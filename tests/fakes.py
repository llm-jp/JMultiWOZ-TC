"""generate.py のテストで共通して使うフェイクオブジェクト。

単体テスト(test_generate.py)と結合テスト(test_generate_integration.py)の
両方から利用する。
"""

import httpx
from openai import APITimeoutError


class FakeFunction:
    """OpenAI SDKの ChatCompletionMessageToolCall.function を模したもの。"""

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    """OpenAI SDKの ChatCompletionMessageToolCall を模したもの。"""

    def __init__(self, name, arguments):
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    """OpenAI SDKの ChatCompletionMessage を模したもの。"""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class FakeChoice:
    """OpenAI SDKの Choice を模したもの。"""

    def __init__(self, message):
        self.message = message


class FakeResponse:
    """OpenAI SDKの ChatCompletion レスポンスを模したもの。"""

    def __init__(self, tool_calls=None):
        self.choices = [FakeChoice(FakeMessage(tool_calls))]


class FakeCompletions:
    """呼び出し回数の記録と、例外/レスポンスの差し替えができるcreate()を提供する。"""

    def __init__(self, responses=None, exceptions=None):
        self._responses = list(responses or [])
        self._exceptions = list(exceptions or [])
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return (
            self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        )


class FakeChat:
    """OpenAI SDKの client.chat を模したもの。"""

    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeClient:
    """OpenAI SDKの OpenAI クライアントを模したもの。"""

    def __init__(self, completions: FakeCompletions):
        self.chat = FakeChat(completions)


def make_timeout_error() -> APITimeoutError:
    """generate.output_with_retries が捕捉する APITimeoutError を生成する。"""
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    return APITimeoutError(request=request)
