from generate import split_batched_tool_calls_in_messages


def test_split_batched_tool_calls_in_messages():
    # 2件のtool_callsがまとまったassistantメッセージ部分。
    messages = [
        {
            "role": "user",
            "content": "CONTENT1",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "ID1",
                    "type": "function",
                    "function": {
                        "name": "TOOL1",
                        "arguments": '{"name": "NAME1", "cashless": "対応", "jumbo": "対応", "city": "CITY1"}',
                    },
                },
                {
                    "id": "ID2",
                    "type": "function",
                    "function": {
                        "name": "TOOL2",
                        "arguments": '{"departurepoint": "DEPATTUREPOINT1", "arrivalpoint": "ARRAIVALPOINT1", "name": "NAME1", "city": "CITY1", "month": 3, "day": 21, "hour": 11, "minute": 30}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "ID1",
            "content": '{"active_entity": {"city": "CITY1", "name": "NAME1", "cashless": "対応", "jumbo": "対応", "phone": "PHONE1", "domain": "DOMAIN1"}}',
        },
        {
            "role": "tool",
            "tool_call_id": "ID2",
            "content": '{"success": false, "ref": null}',
        },
        {
            "role": "assistant",
            "content": "CONTENT1",
        },
    ]

    result = split_batched_tool_calls_in_messages(messages)

    # 2件のtool_callsが1件ずつのassistantメッセージに分割され、
    # 直後に対応するtool結果が交互に並ぶ形式になっていること。
    assert result == [
        {
            "role": "user",
            "content": "CONTENT1",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "ID1",
                    "type": "function",
                    "function": {
                        "name": "TOOL1",
                        "arguments": '{"name": "NAME1", "cashless": "対応", "jumbo": "対応", "city": "CITY1"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "ID1",
            "content": '{"active_entity": {"city": "CITY1", "name": "NAME1", "cashless": "対応", "jumbo": "対応", "phone": "PHONE1", "domain": "DOMAIN1"}}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "ID2",
                    "type": "function",
                    "function": {
                        "name": "TOOL2",
                        "arguments": '{"departurepoint": "DEPATTUREPOINT1", "arrivalpoint": "ARRAIVALPOINT1", "name": "NAME1", "city": "CITY1", "month": 3, "day": 21, "hour": 11, "minute": 30}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "ID2",
            "content": '{"success": false, "ref": null}',
        },
        {
            "role": "assistant",
            "content": "CONTENT1",
        },
    ]