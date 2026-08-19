"""generate.py の結合テスト。

main() を起点に、引数解析・ファイルI/O・LLM呼び出し・出力までの
一連の流れが正しく連携することを確認する。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))
import generate

from fakes import FakeClient, FakeCompletions, FakeResponse, FakeToolCall


def _write_tools_and_input(tmp_path, items):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text("[]", encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
    )
    return tools_path, input_path


def test_main_generates_output_for_each_item(tmp_path, monkeypatch):
    """入力データの全件に対して1回ずつLLMを呼び出し、出力ファイルへ書き出す。"""
    items = [
        {"data_id": "d1", "dialogue_id": "dlg1", "question": [{"role": "user"}]},
        {"data_id": "d2", "dialogue_id": "dlg2", "question": [{"role": "user"}]},
    ]
    tools_path, input_path = _write_tools_and_input(tmp_path, items)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    response = FakeResponse(
        tool_calls=[FakeToolCall("search_hotel", json.dumps({"area": "kyoto"}))]
    )
    completions = FakeCompletions(responses=[response])
    fake_client = FakeClient(completions)
    fake_client.models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="org/model-name")])
    )
    monkeypatch.setattr(generate, "OpenAI", lambda base_url, api_key: fake_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--tools",
            str(tools_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    generate.main()

    output_path = output_dir / "result_org_model-name.jsonl"
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert completions.call_count == 2


def test_main_skips_existing_data_ids(tmp_path, monkeypatch):
    """出力ファイルに既存のdata_idは再実行されず、未処理分のみLLMを呼び出す。"""
    items = [
        {"data_id": "d1", "dialogue_id": "dlg1", "question": [{"role": "user"}]},
        {"data_id": "d2", "dialogue_id": "dlg2", "question": [{"role": "user"}]},
    ]
    tools_path, input_path = _write_tools_and_input(tmp_path, items)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_output = output_dir / "result_org_model-name.jsonl"
    existing_output.write_text(
        json.dumps({"data_id": "d1", "dialogue_id": "dlg1", "tool_calls": []}) + "\n",
        encoding="utf-8",
    )

    response = FakeResponse(tool_calls=[])
    completions = FakeCompletions(responses=[response])
    fake_client = FakeClient(completions)
    fake_client.models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="org/model-name")])
    )
    monkeypatch.setattr(generate, "OpenAI", lambda base_url, api_key: fake_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--tools",
            str(tools_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    generate.main()

    lines = existing_output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert completions.call_count == 1


def test_main_with_batch_size_processes_all_items(tmp_path, monkeypatch):
    """--batch-sizeを1より大きくしても、全件が欠落・重複・行破損なく処理される。"""
    items = [
        {"data_id": f"d{i}", "dialogue_id": f"dlg{i}", "question": [{"role": "user"}]}
        for i in range(10)
    ]
    tools_path, input_path = _write_tools_and_input(tmp_path, items)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    response = FakeResponse(tool_calls=[])
    completions = FakeCompletions(responses=[response])
    fake_client = FakeClient(completions)
    fake_client.models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="org/model-name")])
    )
    monkeypatch.setattr(generate, "OpenAI", lambda base_url, api_key: fake_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--tools",
            str(tools_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "4",
        ],
    )

    generate.main()

    output_path = output_dir / "result_org_model-name.jsonl"
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert completions.call_count == 10
    data_ids = {json.loads(line)["data_id"] for line in lines}
    assert data_ids == {item["data_id"] for item in items}
