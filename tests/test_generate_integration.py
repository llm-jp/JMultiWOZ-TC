"""generate.py の並列実行(threading)に関する結合テスト。

main() を起点に、--batch-size を指定した並列実行でも
欠落・重複・行破損なく処理されることを確認する。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))
import generate

from fakes import FakeClient, FakeCompletions, FakeResponse


def _write_tools_and_input(tmp_path, items):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text("[]", encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
    )
    return tools_path, input_path


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
