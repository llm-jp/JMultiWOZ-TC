import json
import argparse
import httpx
import time
from pathlib import Path
from openai import OpenAI, APITimeoutError


def load_tools(file_path):
    """ツール定義JSONを読み込む

    JMultiWOZを元に定義したツールを格納したJSONファイルを読み込み，その内容を返す．

    Args:
        tools_path (Path): ツールJSONファイルのパス．

    Returns:
        list: ツール定義のリスト．
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl(file_path):
    """JSONLを行単位で読み込む

    JSONLファイルを1行=1レコードとして読み込み、各行のJSONを配列として返す．

    Args:
        file_path (Path): 読み込むJSONLファイルのパス．

    Returns:
        list[dict]: JSON（辞書）のリスト．
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_model_and_safe_name(client) -> tuple[str, str]:
    """モデル名の取得

    起動したvllmサーバーからモデル名を取得し、
    ファイル名に利用できるようスラッシュをアンダースコアへ置換した名称も返す。

    Args:
        client (OpenAI): 使用するOpenAIクライアントインスタンス。

    Returns:
        tuple[str, str]: （モデル名, サニタイズ済みモデル名）。
    """
    model_name = client.models.list().data[0].id
    safe_model_name = model_name.replace("/", "_")
    return model_name, safe_model_name


def load_existing_data_ids(output_path: Path) -> set:
    """出力ファイル作成と既存データIDの収集

    出力JSONLが存在する場合は既存レコードからdata_idを収集し、
    存在しない場合は新規でファイルを作成する。

    Args:
        output_path (Path): 出力JSONLファイルのパス。

    Returns:
        set[str]: 既存のdata_idの集合。
    """
    existing_ids: set[str] = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            if "data_id" in rec:
                                existing_ids.add(rec["data_id"])
                    except Exception:
                        pass
            print(f"既存の出力を検出: {len(existing_ids)}件をスキップして再開します")
        except Exception as e:
            print(f"既存出力の読み取りエラー: {e}")
    else:
        open(output_path, "w").close()
    return existing_ids


def output_with_retries(
    client: OpenAI,
    model_name: str,
    messages,
    tools,
    max_retries: int,
):
    """LLM出力の実行

    タイムアウト(APITimeoutError/httpx.ReadTimeout)時に最大max_retries回まで再試行し、
    最初に成功したレスポンスを返す。

    Args:
        client (OpenAI): 使用するOpenAIクライアントインスタンス。
        model_name (str): 使用するモデル名。
        messages (list[dict]): Chat Completions用のメッセージ配列。
        tools (list): ツール定義。
        max_retries (int): 再試行の最大回数。

    Returns:
        tuple: (response, error) を返す。
        成功時は (response, None)、タイムアウトで失敗時は (None, "TimeoutError")。
    """
    error = None
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            break
        except (APITimeoutError, httpx.ReadTimeout) as e:
            print(f"[Timeout] {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                error = "TimeoutError"
            time.sleep(2)
        except Exception as e:
            print(f"[Error] LLM実行中にエラーが発生しました: {e}")
            error = f"Exception: {e}"
            break
    return response, error


def build_error_record(data_id: str, dialogue_id: str, error: str) -> dict:
    """エラー時のレコード作成

    API呼び出しがエラーした場合に出力するレコード(dict)を生成する。

    Args:
        data_id (str): 入力データのID。
        dialogue_id (str): 対応するダイアログID。
        error (str): エラー内容。

    Returns:
        dict: エラー内容を含むレコード。
    """
    if error == "TimeoutError":
        print("✗ タイムアウトエラーで評価できませんでした。")
    else:
        print(f"✗ エラーで評価できませんでした: {error}")
    return {
        "data_id": data_id,
        "dialogue_id": dialogue_id,
        "tool_calls": [],
        "error": error,
    }


def serialize_tool_calls(tool_calls) -> list:
    """ツール呼び出しをシリアライズ可能に整形

    LLM出力の tool_calls を、後続処理や保存に適した辞書配列へ正規化する。
    function.arguments が文字列の場合はJSONとしてのパースを行う。

    Args:
        tool_calls (list): モデル応答のツール呼び出し配列。

    Returns:
        list: シリアライズ可能な辞書形式のツール呼び出し配列。
    """
    serializable = []
    for tc in tool_calls:
        # 関数呼び出しがある場合のみシリアライズを必要とする
        if hasattr(tc, "function"):
            args_dict = json.loads(tc.function.arguments)
            serializable.append(
                {
                    "name": tc.function.name,
                    "arguments": args_dict,
                }
            )
        else:
            serializable.append(tc)
    return serializable


def log_tool_calls(serializable_tool_calls: list):
    """ツール呼び出し内容のログ出力

    シリアライズ済みツール呼び出し配列を、人が読みやすい形式で出力する。

    Args:
        serializable_tool_calls (list): シリアライズ済みツール呼び出しの配列。

    Returns:
        None: なし。
    """
    print(f"  ツール呼び出し件数: {len(serializable_tool_calls)}")
    if serializable_tool_calls:
        for i, call in enumerate(serializable_tool_calls, 1):
            name = call.get("name")
            args = call.get("arguments")
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_str = str(args)
            print(f"    [{i}] name: {name}")
            print(f"        arguments: {args_str}")
    else:
        print("    ツール呼び出しなし")


def build_success_record(
    data_id: str,
    dialogue_id: str,
    serializable_tool_calls: list,
) -> dict:
    """成功時の最終レコード生成

    ツール呼び出し結果を含む、成功時の出力レコード(dict)を生成する。

    Args:
        data_id (str): 入力データのID。
        dialogue_id (str): 対応するダイアログID。
        serializable_tool_calls (list): シリアライズ済みツール呼び出しの配列。

    Returns:
        dict: 成功時の出力レコード。
    """
    return {
        "data_id": data_id,
        "dialogue_id": dialogue_id,
        "tool_calls": serializable_tool_calls,
    }


def append_jsonl_record(path: Path, record: dict):
    """JSONLへ1レコードを書き出す

    指定された出力ファイルに、UTF-8の1行1JSONとしてレコードを追記する。

    Args:
        path (Path): 出力JSONLファイルのパス。
        record (dict): 追記するレコード内容(辞書形式)。

    Returns:
        None: なし。
    """
    with open(path, "a", encoding="utf-8") as out_f:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_item(
    item: dict,
    tools,
    client,
    model_name: str,
    output_path: Path,
    max_retries: int,
):
    """1レコードを処理して出力まで行う

    1件の入力に対して、LLM出力の実行→ツール呼び出しの整形→出力JSONLへの追記までを行う。

    Args:
        item (dict): 入力レコード。
        tools (list): ツール定義。
        client (OpenAI): 使用する OpenAI クライアントインスタンス。
        model_name (str): 使用するモデル名。
        output_path (Path): 出力JSONLのパス。
        max_retries (int): タイムアウト時の最大再試行回数。

    Returns:
        None: なし。
    """
    data_id = item.get("data_id")
    dialogue_id = item.get("dialogue_id")
    messages = item["question"]

    response, error = output_with_retries(
        client, model_name, messages, tools, max_retries
    )

    if error is not None:
        llm_rec = build_error_record(data_id, dialogue_id, error)
        append_jsonl_record(output_path, llm_rec)
        print("-" * 80)
        return

    tool_calls = (
        response.choices[0].message.tool_calls
        if response.choices[0].message.tool_calls
        else []
    )

    serializable_tool_calls = serialize_tool_calls(tool_calls)
    log_tool_calls(serializable_tool_calls)

    llm_rec = build_success_record(
        data_id,
        dialogue_id,
        serializable_tool_calls,
    )
    append_jsonl_record(output_path, llm_rec)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tools",
        type=Path,
        default="tools.json",
        help="JMultiWOZ-TCに含まれるツールリストのファイルパスを指定",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="jmultiwoz_tc_input.jsonl",
        help="JMultiWOZ-TCに含まれる入力データのファイルパスを指定",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vllmサーバーのベースURLを指定",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="dummy",
        help="OpenAI APIキーを指定",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=".",
        help="出力ファイルを保存するディレクトリを指定",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="タイムアウト時の最大再試行回数を指定",
    )

    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.openai_api_key)

    tools = load_tools(args.tools)
    input_data = load_jsonl(args.input)

    model_name, safe_model_name = get_model_and_safe_name(client)

    output_path = args.output_dir / f"result_{safe_model_name}.jsonl"
    existing_ids = load_existing_data_ids(output_path)

    total = len(input_data)
    print(f"評価開始: {total}件のデータを実行します\n")

    for idx, item in enumerate(input_data, 1):
        data_id = item.get("data_id")
        print(f"[{idx}/{total}] ID: {data_id}")
        if data_id in existing_ids:
            print("→ 既存結果ありのためスキップ")
            print("-" * 80)
            continue

        process_item(
            item,
            tools,
            client,
            model_name,
            output_path,
            args.max_retries,
        )

    print(f"出力結果のJSONLを書き出しました: {output_path}\n")

if __name__ == "__main__":
    main()