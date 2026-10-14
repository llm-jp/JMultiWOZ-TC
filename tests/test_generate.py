"""generate.py の並列実行(threading)に関するテスト。"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import generate

from fakes import FakeClient, FakeCompletions, FakeResponse, FakeToolCall


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
