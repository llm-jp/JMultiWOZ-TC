"""generate.py の各関数に対するテスト。"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
import generate

from fakes import FakeClient, FakeCompletions, FakeResponse, FakeToolCall, make_timeout_error


# --- load_tools ------------------------------------------------------------


def test_load_tools_normal(tmp_path):
    """正常なJSONファイルを読み込むと、内容がそのままリストとして返る。"""
    tools = [{"type": "function", "function": {"name": "search_hotel"}}]
    file_path = tmp_path / "tools.json"
    file_path.write_text(json.dumps(tools), encoding="utf-8")

    result = generate.load_tools(file_path)

    assert result == tools


def test_load_tools_invalid_json_raises(tmp_path):
    """壊れたJSONを読み込むとJSONDecodeErrorが送出される。"""
    file_path = tmp_path / "tools.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        generate.load_tools(file_path)


# --- load_jsonl --------------------------------------------------------------


def test_load_jsonl_normal(tmp_path):
    """複数行のJSONLを読み込むと、各行が辞書のリストとして返る。"""
    records = [{"data_id": "1"}, {"data_id": "2"}]
    file_path = tmp_path / "input.jsonl"
    file_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    result = generate.load_jsonl(file_path)

    assert result == records


def test_load_jsonl_empty_file_returns_empty_list(tmp_path):
    """空ファイルを読み込むと空リストが返る。"""
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("", encoding="utf-8")

    result = generate.load_jsonl(file_path)

    assert result == []


# --- get_model_and_save_name -------------------------------------------------


def test_get_model_and_save_name_normal():
    """スラッシュを含むモデル名は、ファイル名用にアンダースコアへ置換される。"""
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="org/model-name")])
        )
    )

    model_name, save_model_name = generate.get_model_and_save_name(client)

    assert model_name == "org/model-name"
    assert save_model_name == "org_model-name"


def test_get_model_and_save_name_without_slash(tmp_path):
    """スラッシュを含まないモデル名は、そのまま保存用名称として使われる。"""
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="model")])
        )
    )

    model_name, save_model_name = generate.get_model_and_save_name(client)

    assert model_name == "model"
    assert save_model_name == "model"


# --- load_existing_data_ids --------------------------------------------------


def test_load_existing_data_ids_normal(tmp_path):
    """既存の出力JSONLから、すべてのdata_idが集合として取得できる。"""
    output_path = tmp_path / "result.jsonl"
    output_path.write_text(
        json.dumps({"data_id": "1"}) + "\n" + json.dumps({"data_id": "2"}) + "\n",
        encoding="utf-8",
    )

    result = generate.load_existing_data_ids(output_path)

    assert result == {"1", "2"}


def test_load_existing_data_ids_missing_file_creates_empty_file(tmp_path):
    """出力ファイルが存在しない場合、空ファイルが新規作成され空集合が返る。"""
    output_path = tmp_path / "result.jsonl"

    result = generate.load_existing_data_ids(output_path)

    assert result == set()
    assert output_path.exists()


def test_load_existing_data_ids_skips_corrupted_line(tmp_path):
    """壊れた行が混在していても、例外を送出せず正常な行のみを収集する。"""
    output_path = tmp_path / "result.jsonl"
    output_path.write_text(
        json.dumps({"data_id": "1"}) + "\n" + "{not valid json\n",
        encoding="utf-8",
    )

    result = generate.load_existing_data_ids(output_path)

    assert result == {"1"}


# --- output_with_retries -----------------------------------------------------


def test_output_with_retries_success_on_first_attempt():
    """1回目の呼び出しで成功すれば、レスポンスがそのまま返りエラーはNoneになる。"""
    response = FakeResponse()
    client = FakeClient(FakeCompletions(responses=[response]))

    result, error = generate.output_with_retries(
        client, "model", [{"role": "user", "content": "hi"}], [], max_retries=3
    )

    assert result is response
    assert error is None


def test_output_with_retries_timeout_exhausted(monkeypatch):
    """max_retries回すべてタイムアウトすると、"TimeoutError"を返して再試行を打ち切る。"""
    monkeypatch.setattr(generate.time, "sleep", lambda seconds: None)
    completions = FakeCompletions(
        exceptions=[make_timeout_error(), make_timeout_error(), make_timeout_error()]
    )
    client = FakeClient(completions)

    result, error = generate.output_with_retries(
        client, "model", [], [], max_retries=3
    )

    assert result is None
    assert error == "TimeoutError"
    assert completions.call_count == 3


def test_output_with_retries_generic_exception_breaks_immediately(monkeypatch):
    """タイムアウト以外の例外は再試行せず、1回で打ち切ってエラー内容を返す。"""
    monkeypatch.setattr(generate.time, "sleep", lambda seconds: None)
    completions = FakeCompletions(exceptions=[ValueError("boom")])
    client = FakeClient(completions)

    result, error = generate.output_with_retries(
        client, "model", [], [], max_retries=3
    )

    assert result is None
    assert error == "Exception: boom"
    assert completions.call_count == 1


# --- build_error_record -------------------------------------------------------


def test_build_error_record_timeout(capsys):
    """TimeoutErrorの場合、専用メッセージを出力しつつエラーレコードを生成する。"""
    record = generate.build_error_record("d1", "dlg1", "TimeoutError")

    assert record == {
        "data_id": "d1",
        "dialogue_id": "dlg1",
        "tool_calls": [],
        "error": "TimeoutError",
    }
    assert "タイムアウト" in capsys.readouterr().out


def test_build_error_record_generic_exception(capsys):
    """タイムアウト以外のエラーの場合、汎用メッセージを出力しつつエラーレコードを生成する。"""
    record = generate.build_error_record("d1", "dlg1", "Exception: boom")

    assert record["error"] == "Exception: boom"
    assert "boom" in capsys.readouterr().out


# --- serialize_tool_calls ------------------------------------------------------


def test_serialize_tool_calls_normal():
    """function属性を持つツール呼び出しは、name/argumentsの辞書へ整形される。"""
    tool_calls = [FakeToolCall("search_hotel", json.dumps({"area": "kyoto"}))]

    result = generate.serialize_tool_calls(tool_calls)

    assert result == [{"name": "search_hotel", "arguments": {"area": "kyoto"}}]


def test_serialize_tool_calls_non_dict_arguments_raises():
    """argumentsがdict型にパースできない場合、ValueErrorが送出される。"""
    tool_calls = [FakeToolCall("search_hotel", json.dumps([1, 2, 3]))]

    with pytest.raises(ValueError):
        generate.serialize_tool_calls(tool_calls)


def test_serialize_tool_calls_without_function_attribute_passthrough():
    """function属性を持たない要素(整形済みデータ)は、そのまま素通りする。"""
    already_serialized = {"name": "search_hotel", "arguments": {"area": "kyoto"}}

    result = generate.serialize_tool_calls([already_serialized])

    assert result == [already_serialized]


# --- log_tool_calls -------------------------------------------------------------


def test_log_tool_calls_with_calls(capsys):
    """ツール呼び出しがある場合、件数と各呼び出しの内容がログに出力される。"""
    generate.log_tool_calls([{"name": "search_hotel", "arguments": {"area": "kyoto"}}])

    out = capsys.readouterr().out
    assert "ツール呼び出し件数: 1" in out
    assert "search_hotel" in out


def test_log_tool_calls_empty(capsys):
    """ツール呼び出しがない場合、「呼び出しなし」のメッセージが出力される。"""
    generate.log_tool_calls([])

    assert "ツール呼び出しなし" in capsys.readouterr().out


# --- build_success_record -------------------------------------------------------


def test_build_success_record_normal():
    """ツール呼び出しを含む成功レコードが、期待した形式で生成される。"""
    record = generate.build_success_record(
        "d1", "dlg1", [{"name": "search_hotel", "arguments": {}}]
    )

    assert record == {
        "data_id": "d1",
        "dialogue_id": "dlg1",
        "tool_calls": [{"name": "search_hotel", "arguments": {}}],
    }


def test_build_success_record_empty_tool_calls():
    """ツール呼び出しが空でも、tool_callsが空リストのレコードが生成される。"""
    record = generate.build_success_record("d1", "dlg1", [])

    assert record["tool_calls"] == []


# --- append_jsonl_record ----------------------------------------------------------


def test_append_jsonl_record_normal(tmp_path):
    """レコードを書き出すと、1行のJSONとしてファイルに保存される。"""
    path = tmp_path / "out.jsonl"
    record = {"data_id": "1"}

    generate.append_jsonl_record(path, record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_append_jsonl_record_appends_without_overwriting(tmp_path):
    """複数回書き出すと、既存の行を上書きせず末尾に追記される。"""
    path = tmp_path / "out.jsonl"

    generate.append_jsonl_record(path, {"data_id": "1"})
    generate.append_jsonl_record(path, {"data_id": "2"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["data_id"] for line in lines] == ["1", "2"]


# --- process_item ----------------------------------------------------------------


def test_process_item_success(tmp_path):
    """LLM呼び出しが成功すると、ツール呼び出し内容を含む成功レコードが出力される。"""
    output_path = tmp_path / "out.jsonl"
    response = FakeResponse(
        tool_calls=[FakeToolCall("search_hotel", json.dumps({"area": "kyoto"}))]
    )
    client = FakeClient(FakeCompletions(responses=[response]))
    item = {"data_id": "d1", "dialogue_id": "dlg1", "question": [{"role": "user"}]}

    generate.process_item(
        item, [], client, "model", output_path, max_retries=3, write_lock=threading.Lock()
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["data_id"] == "d1"
    assert record["tool_calls"] == [{"name": "search_hotel", "arguments": {"area": "kyoto"}}]


def test_process_item_timeout_error_writes_error_record(tmp_path, monkeypatch):
    """LLM呼び出しがタイムアウトすると、tool_callsが空のエラーレコードが出力される。"""
    monkeypatch.setattr(generate.time, "sleep", lambda seconds: None)
    output_path = tmp_path / "out.jsonl"
    client = FakeClient(FakeCompletions(exceptions=[make_timeout_error()]))
    item = {"data_id": "d1", "dialogue_id": "dlg1", "question": [{"role": "user"}]}

    generate.process_item(
        item, [], client, "model", output_path, max_retries=1, write_lock=threading.Lock()
    )

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["error"] == "TimeoutError"
    assert record["tool_calls"] == []


def test_process_item_concurrent_writes_are_not_corrupted(tmp_path):
    """複数スレッドから同時に呼び出しても、write_lockにより出力行が壊れず全件書き込まれる。"""
    output_path = tmp_path / "out.jsonl"
    response = FakeResponse(
        tool_calls=[FakeToolCall("search_hotel", json.dumps({"area": "kyoto"}))]
    )
    write_lock = threading.Lock()
    items = [
        {"data_id": f"d{i}", "dialogue_id": f"dlg{i}", "question": [{"role": "user"}]}
        for i in range(20)
    ]

    def run(item):
        client = FakeClient(FakeCompletions(responses=[response]))
        generate.process_item(
            item, [], client, "model", output_path, max_retries=3, write_lock=write_lock
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run, items))

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    data_ids = {json.loads(line)["data_id"] for line in lines}
    assert data_ids == {item["data_id"] for item in items}
