def split_batched_tool_calls_in_messages(messages: list) -> list:
    """1メッセージにまとめたtool_callsを1呼び出しずつの交互形式に変換する

    [a・b呼び出しをまとめた1つのassistantメッセージ]→[aの結果]→[bの結果] という形式を、
    [a呼び出し]→[aの結果]→[b呼び出し]→[bの結果] の順に1件ずつ交互に並ぶ形式へ変換する。
    1メッセージ内の複数tool_callsに対応していないモデル向けのフォールバックとして使用する。

    Args:
        messages (list[dict]): 変換対象のメッセージ配列。

    Returns:
        list[dict]: 変換後のメッセージ配列。
    """
    results_by_id = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            results_by_id[m.get("tool_call_id")] = m
    consumed_ids = set()
    new_messages = []
    for m in messages:
        tool_calls = m.get("tool_calls") if isinstance(m, dict) else None
        is_batched_call = (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and isinstance(tool_calls, list)
            and len(tool_calls) > 1
        )
        if is_batched_call:
            for tc in tool_calls:
                new_messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tc]}
                )
                call_id = tc.get("id")
                res = results_by_id.get(call_id)
                if res is not None:
                    new_messages.append(res)
                    consumed_ids.add(call_id)
        elif (
            isinstance(m, dict)
            and m.get("role") == "tool"
            and m.get("tool_call_id") in consumed_ids
        ):
            continue
        else:
            new_messages.append(m)
    return new_messages


def test_split_batched_tool_calls_in_messages():
    # 2件のtool_callsがまとまったassistantメッセージ部分。
    messages = [
        {
            "role": "user",
            "content": "1",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "Search_taxi",
                        "arguments": '{"name": "1", "cashless": "対応", "jumbo": "対応", "city": "1"}',
                    },
                },
                {
                    "id": "2",
                    "type": "function",
                    "function": {
                        "name": "Booking_taxi",
                        "arguments": '{"departurepoint": "1", "arrivalpoint": "1", "name": "1", "city": "1", "month": 3, "day": 21, "hour": 11, "minute": 30}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": '{"active_entity": {"city": "1", "name": "1", "cashless": "対応", "jumbo": "対応", "phone": "1", "domain": "taxi"}}',
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "content": '{"success": false, "ref": null}',
        },
        {
            "role": "assistant",
            "content": "1",
        },
    ]

    result = split_batched_tool_calls_in_messages(messages)

    # 2件のtool_callsが1件ずつのassistantメッセージに分割され、
    # 直後に対応するtool結果が交互に並ぶ形式になっていること。
    assert result == [
        {
            "role": "user",
            "content": "1",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "Search_taxi",
                        "arguments": '{"name": "1", "cashless": "対応", "jumbo": "対応", "city": "1"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": '{"active_entity": {"city": "1", "name": "1", "cashless": "対応", "jumbo": "対応", "phone": "1", "domain": "taxi"}}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {
                        "name": "Booking_taxi",
                        "arguments": '{"departurepoint": "1", "arrivalpoint": "1", "name": "1", "city": "1", "month": 3, "day": 21, "hour": 11, "minute": 30}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "content": '{"success": false, "ref": null}',
        },
        {
            "role": "assistant",
            "content": "1",
        },
    ]