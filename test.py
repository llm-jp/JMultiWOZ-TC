from openai import OpenAI
import json
from pathlib import Path

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [
    {
        "type": "function",
        "function": {"name": "Search_restaurant", "description": "訪れたいレストランを検索し、地域・営業時間・価格帯などの詳細情報を提示します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "レストランを見つけたい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "レストランの名前"}, "genre": {"type": "string", "description": "レストランの種類('居酒屋','和食','寿司・魚料理','うどん・そば','鍋','お好み焼き・たこ焼き','ラーメン・つけ麺','郷土料理','洋食・西洋料理','カレー','焼肉・ホルモン','イタリアン・フレンチ','中華料理','アジア・エスニック料理','カフェ・スイーツ','ホテルレストラン','その他')"}, "area": {"type": "string", "description": "レストランがある地域"}, "pricerange": {"type": "string", "description": "レストランの価格帯(安め、普通、高め)"}, "station": {"type": "string", "description": "最寄りの鉄道または地下鉄の駅"}, "wifi": {"type": "string", "description": "レストランがWi-Fiを提供しているかどうか（'有り(無料)','有り(有料)','無し','不明'）"}, "parking": {"type": "string", "description": "レストランが駐車場を提供しているかどうか（'有り(無料)','有り(有料)','無し','不明'）"}}, "required": ["city"]}},
    },{
        "type": "function",
        "function": {"name": "Search_hotel", "description": "ユーザーの希望条件（地域、価格、設備など）をもとに宿泊施設を探し、候補となるホテルや旅館を紹介します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "ホテルを探したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "ホテルの名前"}, "genre": {"type": "string", "description": "ホテルの種類('旅館','リゾートホテル','ビジネスホテル','シティホテル','簡易宿所(ベッドハウス・山小屋・カプセルホテル等)','その他')"}, "area": {"type": "string", "description": "ホテルがある地域"}, "pricerange": {"type": "string", "description": "ホテルの価格帯(安め、普通、高め)"}, "station": {"type": "string", "description": "最寄りの鉄道または地下鉄の駅"}, "wifi": {"type": "string", "description": "ホテルがWi-Fiを提供しているかどうか（'有り(無料)','有り(有料)','無し','不明'）"}, "parking": {"type": "string", "description": "ホテルが駐車施設を提供しているかどうか（'有り(無料)','有り(有料)','無し','不明'）"}, "withrestaurant": {"type": "string", "description": "ホテルにレストランがあるかどうか（'有り','無し','不明'）"}}, "required": ["city"]}},
    },{
        "type": "function",
        "function": {"name": "Search_attraction", "description": "観光地や名所、体験スポットなどを条件に応じて提案し、アクセス情報や営業時間などもあわせて案内します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "観光施設を探したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "観光施設の名前"}, "genre": {"type": "string", "description": "観光施設のジャンル('科学館','展望台・タワー','伝統芸能','銭湯・スパ','温泉','プラネタリウム','絶景','伝統文化体験','リゾート・保養地','美術館・博物館','散策エリア','教室・ワークショップ・体験','建築物','歴史的建造物','城','歴史的な散策エリア','神社・寺院・教会・モニュメントなど','庭園','水族館','動物園','植物園','都市の景観','工業地域','工場見学','産業観光','自然・公園','牧場','アリーナ・スタジアム','映画館','橋','コンサート・ショー','エンタメ・アミューズメント','記念碑・像','旧跡','テーマパーク','港・波止場','フェリー','アウトドア','スポーツ・フィットネス','桜','アートギャラリー','ビーチ','キャンプ','グランピング','世界遺産','その他')"}, "area": {"type": "string", "description": "観光施設がある地域"}, "station": {"type": "string", "description": "観光施設の最寄りの鉄道または地下鉄の駅"}, "wifi": {"type": "string", "description": "観光施設にWi-Fiが提供されているかどうか（'有り(無料)','有り(有料)','無し','不明'）"}, "parking": {"type": "string", "description": "観光地に駐車場があるかどうか（'有り(無料)','有り(有料)','無し','不明'）"}},  "required": ["city"]}},
    },{
        "type": "function",
        "function": {"name": "Search_shopping", "description": "地域やカテゴリに応じてショッピング施設を検索し、店舗情報や営業時間などを紹介します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "ショッピング施設を探したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "ショッピング施設の名前"}, "genre": {"type": "string", "description": "ショッピング施設のジャンル('お土産','アウトレットモール','ショッピングモール','商店街','百貨店や総合スーパー','スーパーマーケット','コンビニ','食料品','書籍','文房具・雑貨','ファッション','おもちゃ','スポーツ','医薬品・化粧品','家電・電化製品','旅行用品','自動車・自転車','燃料','その他')"}, "area": {"type": "string", "description": "ショッピング施設がある地域"}, "station": {"type": "string", "description": "最寄りの鉄道または地下鉄の駅"}, "parking": {"type": "string", "description": "ショッピング施設に駐車場があるかどうか（'有り(無料)','有り(有料)','無し','不明'）"}},  "required": ["city"]}},
    },{
        "type": "function",
        "function": {"name": "Search_taxi", "description": "希望の地域の周辺で利用可能なタクシー会社を案内し、支払い方法などの情報を提供します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "タクシーを利用したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "タクシー会社の名前"}, "cashless": {"type": "string", "description": "タクシー会社がキャッシュレス決済に対応しているかどうか（'対応','非対応','不明'）"}, "jumbo": {"type": "string", "description": "ジャンボタクシーに対応しているかどうか（'対応','非対応','不明'）"}},  "required": ["city"]}},
    },{
        "type": "function",
        "function": {"name": "Search_Weather", "description": "指定された地域と日付に応じて、天気予報、最高/最低気温を表示します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "天気情報を確認したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "area": {"type": "string", "description": "天気情報を確認したい地域"}, "month": {"type": "integer", "description": "天気情報を確認したい月"}, "day": {"type": "integer", "description": "天気情報を確認したい日"}},  "required": ["city"]}},
    },{
        "type": "function",
        "function": {"name": "Booking_restaurant", "description": "希望する日時や人数、条件に基づいて、対象のレストランをオンラインで予約できるようにします。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "レストランを予約したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "予約したいレストランの名前"}, "people": {"type": "integer", "description": "予約する人数"}, "month": {"type": "integer", "description": "予約する月"}, "day": {"type": "integer", "description": "予約する日"}, "hour": {"type": "integer", "description": "予約する時間（時）"}, "minute": {"type": "integer", "description": "予約する時間（分）"}}, "required": ["city", "name", "people", "month", "day", "hour", "minute"]}},
    },{
        "type": "function",
        "function": {"name": "Booking_hotel", "description": "希望した宿泊施設の空室状況を確認し、そのまま予約を完了できるよう支援します。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "ホテルを予約したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "予約したいホテルの名前"}, "people": {"type": "integer", "description": "ホテルの部屋に宿泊する人数"}, "month": {"type": "integer", "description": "ホテルの宿泊月"}, "day": {"type": "integer", "description": "ホテルの宿泊日"}, "stay": {"type": "integer", "description": "ホテルの宿泊日数"}}, "required": ["city", "name", "people", "month", "day", "stay"]}},
    },{
        "type": "function",
        "function": {"name": "Booking_taxi", "description": "出発地と目的地、利用時間をもとにタクシーの予約を代行します。タクシーの手配を効率的に行うことができます。", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "タクシーを予約したい都市('札幌','仙台','東京','横浜','名古屋','京都','大阪','福岡','那覇')"}, "name": {"type": "string", "description": "予約したいタクシーの名前"}, "month": {"type": "integer", "description": "タクシー利用月"}, "day": {"type": "integer", "description": "タクシー利用日"}, "hour": {"type": "integer", "description": "タクシー利用時間（時）"}, "minute": {"type": "integer", "description": "タクシー利用時間（分）"}, "departurepoint": {"type": "string", "description": "タクシーの乗車場所"}, "arrivalpoint": {"type": "string", "description": "タクシーの降車場所"}}, "required": ["city", "name", "month", "day", "hour", "minute", "departurepoint", "arrivalpoint"]}},
    }
]

def normalize_tool_calls(tool_calls):
    # ツール呼び出しを正規化して比較可能な形式に変換
    if not tool_calls:
        return set()
    
    normalized = []
    for call in tool_calls:
        # LLMの返す tool_call
        if hasattr(call, "function"):
            name = call.function.name
            args_dict = json.loads(call.function.arguments)
        else:
            # goldの dict 形式
            name = call["name"]
            args_dict = call["arguments"]

        args = tuple(sorted(args_dict.items()))
        normalized.append((name, args))
    return set(normalized)

def load_jsonl(file_path):
    # JSONファイルを読み込む
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# データファイルの読み込み
input_file = Path("tool_use_ja_input.json")
ground_file = Path("tool_use_ja_ground.json")

input_data = load_jsonl(input_file)
ground_data = load_jsonl(ground_file)

# 実行モデル名の取得（先頭のモデルを使用）
model_name = client.models.list().data[0].id
# ファイル名に使えるようにサニタイズ（例: "Qwen/Qwen3-14B" -> "Qwen_Qwen3-14B"）
safe_model_name = model_name.replace("/", "_")

# 正解データをidでマッピング
ground_truth_map = {item["id"]: item["ground_truth"] for item in ground_data}

# 統計情報
total = len(input_data)
correct = 0
incorrect = 0
correct_with_output = []  # 正解で出力があったケースを記録
correct_with_nonempty_gold = 0  # 正解のうち ground_truth が空でない件数
incorrect_outputs = []  # 不正解のJSON出力用に記録
all_llm_outputs = []  # 全ケースのLLM出力（idとtool_callsのみ）を記録

# 追加の集計（ツール使用/不使用別）
tool_used_total = 0
tool_used_correct = 0
tool_used_incorrect = 0
tool_unused_total = 0
tool_unused_correct = 0
tool_unused_incorrect = 0

print(f"評価開始: {total}件のデータを処理します\n")

# 各データに対して評価
for idx, item in enumerate(input_data, 1):
    data_id = item["id"]
    messages = item["question"]
    
    print(f"[{idx}/{total}] ID: {data_id}")
    # print(f"メッセージ: {messages}")
    
    # LLMにリクエスト
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    # ツール呼び出しの取得
    tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []
    
    # 正解データの取得
    gold = ground_truth_map.get(data_id, [])
    
    # 正規化して比較
    predicted = normalize_tool_calls(tool_calls)
    expected = normalize_tool_calls(gold)
    
    is_tool_used_case = bool(gold)  # ground_truth が空でない場合はツール使用ケース
    if is_tool_used_case:
        tool_used_total += 1
    else:
        tool_unused_total += 1

    if predicted == expected:
        print("✓ 正解")
        print(f"  予測: {tool_calls}")
        print(f"  正解: {gold}")
        correct += 1
        if is_tool_used_case:
            tool_used_correct += 1
        else:
            tool_unused_correct += 1
        # LLMの出力が空でない場合は記録
        if tool_calls:
            correct_with_output.append({
                "id": data_id,
                "question": messages,
                "predicted": tool_calls,
                "gold": gold
            })
        # ground_truth が空でない正解数のカウント
        if gold:
            correct_with_nonempty_gold += 1
    else:
        print("✗ 不正解")
        print(f"  予測: {response.choices[0].message}")
        print(f"  正解: {gold}")
        incorrect += 1
        if is_tool_used_case:
            tool_used_incorrect += 1
        else:
            tool_unused_incorrect += 1
        # 不正解の出力をNDJSON用に整形して記録
        serializable_tool_calls = []
        for tc in tool_calls:
            if hasattr(tc, "function"):
                # OpenAI SDKのToolCallオブジェクトをシリアライズ可能な辞書へ
                try:
                    args_dict = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except Exception:
                    args_dict = tc.function.arguments
                serializable_tool_calls.append({
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": args_dict,
                    },
                })
            else:
                # 既に辞書形式（goldと同形式）の場合
                serializable_tool_calls.append(tc)

        incorrect_outputs.append({
            "id": data_id,
            "tool_calls": serializable_tool_calls,
            "gold": gold,
        })
    
    # 正解・不正解に関わらず、LLMのtool_callsをシリアライズして全件記録
    serializable_tool_calls_all = []
    for tc in tool_calls:
        if hasattr(tc, "function"):
            try:
                args_dict = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except Exception:
                args_dict = tc.function.arguments
            serializable_tool_calls_all.append({
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": args_dict,
                },
            })
        else:
            serializable_tool_calls_all.append(tc)
    all_llm_outputs.append({
        "id": data_id,
        "tool_calls": serializable_tool_calls_all,
    })
    
    print("-" * 80)

# 結果サマリー
print(f"\n{'='*80}")
print(f"評価結果")
print(f"{'='*80}")
print(f"総データ数: {total}")
print(f"正解: {correct} ({correct/total*100:.2f}%)")
print(f"不正解: {incorrect} ({incorrect/total*100:.2f}%)")
print(f"{'='*80}")

# 不正解の出力をNDJSONで書き出し（先頭行はサマリー）
output_ndjson_path = Path(f"score_{safe_model_name}.json")
with open(output_ndjson_path, "w", encoding="utf-8") as out_f:
    # サマリー1行目
    # {データの合計数、正解数、不正解数、ツール使用時の正解数、ツール使用時の不正解数、ツール不使用時の正解数、ツール不使用時の不正解数、全体の正答率、ツール使用時の正答率、ツール不使用時の正答率}
    overall_acc = (correct / total * 100) if total else 0.0
    tool_used_acc = (tool_used_correct / tool_used_total * 100) if tool_used_total else 0.0
    tool_unused_acc = (tool_unused_correct / tool_unused_total * 100) if tool_unused_total else 0.0
    summary = {
        "データ合計数": total,
        "正解数": correct,
        "不正解数": incorrect,
        "全体の正答率(%)": round(overall_acc, 2),
        "ツール使用時のデータ数": tool_used_total, 
        "ツール使用時の正解数": tool_used_correct,
        "ツール使用時の不正解数": tool_used_incorrect,
        "ツール使用時の正答率(%)": round(tool_used_acc, 2),
        "ツール不使用時のデータ数": tool_unused_total,
        "ツール不使用時の正解数": tool_unused_correct,
        "ツール不使用時の不正解数": tool_unused_incorrect,
        "ツール不使用時の正答率(%)": round(tool_unused_acc, 2),
    }
    out_f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    # 以降は不正解の各ケース
    for rec in incorrect_outputs:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"評価結果のJSONを書き出しました: {output_ndjson_path}")

# LLM出力のみを書き出す別ファイルのJSON
llm_output_ndjson_path = Path(f"result_{safe_model_name}.json")
with open(llm_output_ndjson_path, "w", encoding="utf-8") as out_f:
    for rec in all_llm_outputs:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"LLM出力のみのJSONを書き出しました: {llm_output_ndjson_path}")

# 正解でLLM出力があったケースの詳細表示
# if correct_with_output:
#     print(f"\n{'='*80}")
#     print(f"正解でLLMが出力したケース: {len(correct_with_output)}件")
#     print(f"{'='*80}\n")
    
#     for idx, case in enumerate(correct_with_output, 1):
#         print(f"[{idx}/{len(correct_with_output)}] ID: {case['id']}")
#         # print(f"質問: {case['question']}")
#         print(f"予測: {case['predicted']}")
#         print(f"正解: {case['gold']}")
#         print("-" * 80)