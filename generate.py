import json
from pathlib import Path
import time
from openai import OpenAI, APITimeoutError
import httpx

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
MAX_RETRIES = 3

# ツール定義を外部ファイルから読み込み
TOOLS_PATH = Path("tools.json")
with open(TOOLS_PATH, "r", encoding="utf-8") as f:
    tools = json.load(f)


def load_jsonl(file_path):
    # JSONファイルを読み込む
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# データファイルの読み込み
input_file = Path("jmultiwoz_tc_input.json")
ground_file = Path("jmultiwoz_tc_ground.json")

input_data = load_jsonl(input_file)

# 実行モデル名の取得（先頭のモデルを使用）
model_name = client.models.list().data[0].id
# ファイル名に使えるようにサニタイズ（例: "Qwen/Qwen3-14B" -> "Qwen_Qwen3-14B"）
safe_model_name = model_name.replace("/", "_")

# LLM出力（逐次書き出し）の出力先を初期化
llm_output_ndjson_path = Path(f"result_{safe_model_name}.json")
existing_ids = set()
if llm_output_ndjson_path.exists():
    try:
        with open(llm_output_ndjson_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        # 後方互換: 旧フォーマット(id)と新フォーマット(data_id)の両方に対応
                        if "data_id" in rec:
                            existing_ids.add(rec["data_id"])
                        elif "id" in rec:
                            existing_ids.add(rec["id"])
                except Exception:
                    # 読み取り不能な行はスキップ
                    pass
        print(f"既存の出力を検出: {len(existing_ids)}件をスキップして再開します")
    except Exception as e:
        print(f"既存出力の読み取りでエラー: {e} -> 新規として実行します")
else:
    # 出力ファイルが存在しなければ作成
    open(llm_output_ndjson_path, "w").close()

total = len(input_data)

print(f"評価開始: {total}件のデータを実行します\n")

# 各データに対して評価
for idx, item in enumerate(input_data, 1):
    # 新フォーマットでは data_id 必須、旧フォーマット(id)にも後方互換
    data_id = item.get("data_id") if isinstance(item, dict) else None
    if data_id is None:
        data_id = item["id"]
    dialogue_id = item.get("dialogue_id") if isinstance(item, dict) else None
    messages = item["question"]

    print(f"[{idx}/{total}] ID: {data_id}")
    # print(f"メッセージ: {messages}")
    # 途中再開: 既に出力済みIDはスキップ
    if data_id in existing_ids:
        print("→ 既存結果ありのためスキップ")
        print("-" * 80)
        continue

    error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # LLMにリクエスト
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            break  # 成功したらループを抜ける
        except (APITimeoutError, httpx.ReadTimeout) as e:
            print(f"[Timeout] {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                error = "TimeoutError"
            time.sleep(2)  # 少し待って再試行

    if error == "TimeoutError":
        print(f"✗ タイムアウトエラーで評価できませんでした。")
        llm_rec = {
            "data_id": data_id,
            "dialogue_id": dialogue_id,
            "tool_calls": [],
            "error": "TimeoutError",
        }
        with open(llm_output_ndjson_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(llm_rec, ensure_ascii=False) + "\n")
        print("-" * 80)
        continue

    else:
        # ツール呼び出しの取得
        tool_calls = (
            response.choices[0].message.tool_calls
            if response.choices[0].message.tool_calls
            else []
        )

        serializable_tool_calls_all = []
        for tc in tool_calls:
            if hasattr(tc, "function"):
                try:
                    args_dict = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except Exception:
                    args_dict = tc.function.arguments
                serializable_tool_calls_all.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": args_dict,
                        },
                    }
                )
            else:
                serializable_tool_calls_all.append(tc)

        # 成功時: function callingの中身を出力
        print(f"  ツール呼び出し件数: {len(serializable_tool_calls_all)}")
        if serializable_tool_calls_all:
            for i, call in enumerate(serializable_tool_calls_all, 1):
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name")
                args = fn.get("arguments")
                try:
                    args_str = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args_str = str(args)
                print(f"    [{i}] name: {name}")
                print(f"        arguments: {args_str}")
        else:
            print("    ツール呼び出しなし")
        llm_rec = {
            "data_id": data_id,
            "dialogue_id": dialogue_id,
            "tool_calls": serializable_tool_calls_all,
        }
        # 1件ずつ追記
        with open(llm_output_ndjson_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(llm_rec, ensure_ascii=False) + "\n")

        print("-" * 80)


print(f"出力結果のJSONを書き出しました: {llm_output_ndjson_path}\n")
