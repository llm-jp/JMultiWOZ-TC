import json
import argparse
import re
from pathlib import Path
import random
import csv

# --result resutt_{model-name}.json


def load_jsonl(file_path: Path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def normalize_tool_calls(tool_calls):
    if not tool_calls:
        return set()

    def _canonicalize_arguments(args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        try:
            if isinstance(args, (dict, list)):
                return json.dumps(args, sort_keys=True, ensure_ascii=False)
            else:
                return json.dumps(args, ensure_ascii=False)
        except Exception:
            return str(args)

    normalized = []
    for call in tool_calls:
        if hasattr(call, "function"):
            # OpenAI SDKのToolCallオブジェクト
            name = call.function.name
            raw_args = call.function.arguments
        elif isinstance(call, dict):
            # dict形式: 入れ子/トップレベルの両方に対応
            if "function" in call and isinstance(call["function"], dict):
                fn = call["function"]
                name = fn.get("name")
                raw_args = fn.get("arguments")
            else:
                # トップレベルに name/arguments があるケース
                name = call.get("name")
                raw_args = call.get("arguments")
        else:
            # 不明形式は文字列化して比較可能にする
            name = str(call)
            raw_args = None

        args_canonical = _canonicalize_arguments(raw_args)
        normalized.append((name, args_canonical))

    return set(normalized)


def main():
    parser = argparse.ArgumentParser(description="LLM出力と正解データを比較し、評価結果を出力します。正解データは固定: tool_use_ja_ground.json")
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="LLM出力ファイル(result_{safe_model_name}.json: NDJSON)のパス",
    )
    args = parser.parse_args()

    # 入力(ユーザ質問)のロード: 大規模でも逐次読み込みでマップ化
    input_path = Path("tool_use_ja_input.json")
    question_map = {}
    dialogue_map = {}
    if input_path.exists():
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # 後方互換: 旧フォーマット(id)と新フォーマット(data_id)
                        qid = obj.get("data_id")
                        if qid is None:
                            qid = obj.get("id")
                        if qid is not None:
                            question_map[qid] = obj.get("question")
                            dialogue_map[qid] = obj.get("dialogue_id")
                    except Exception:
                        # 壊れた行はスキップ
                        continue
        except Exception:
            question_map = {}
            dialogue_map = {}

    result_data = load_jsonl(args.result)
    # 正解データは固定
    ground_data = load_jsonl(Path("tool_use_ja_ground.json"))

    # 後方互換: groundのキーを data_id 優先で取り出す
    ground_truth_map = {}
    for item in ground_data:
        if not isinstance(item, dict):
            continue
        key = item.get("data_id")
        if key is None:
            key = item.get("id")
        if key is not None:
            ground_truth_map[key] = item.get("ground_truth", [])

    # 集計用メトリクス
    total = len(result_data)

    # 全体(厳密一致)
    overall_error = 0
    overall_correct = 0
    overall_incorrect = 0

    # ツール使用判断(ゴールドあり -> ツール呼び出し有無のみ判定)
    use_total = 0
    use_error = 0
    use_correct = 0
    use_incorrect = 0

    # ツール不使用判断(ゴールドなし -> ツール未呼び出しを正)
    nouse_total = 0
    nouse_error = 0
    nouse_correct = 0
    nouse_incorrect = 0

    # ツール使用・不使用判断(上記2つの合算: bool(gold) == bool(pred))
    use_or_nouse_error = 0
    use_or_nouse_correct = 0
    use_or_nouse_incorrect = 0

    # tool call精度(ゴールドあり のみで厳密比較)
    call_total = 0  # 対象データ数(= goldあり のみ; 出力ミスは除外)
    call_correct = 0
    call_incorrect = 0

    # 誤答ログ
    incorrect_call_precision = []      # tool call精度の不正解
    incorrect_use_judgement = []       # ツール使用判断の不正解
    incorrect_nouse_judgement = []     # ツール不使用判断の不正解

    print(f"評価開始: {total}件のデータを処理します\n")

    for idx, rec in enumerate(result_data, 1):
        # 後方互換: 出力のキーも data_id 優先
        data_id = rec.get("data_id")
        if data_id is None:
            data_id = rec.get("id")
        predicted_calls = rec.get("tool_calls", [])
        gold = ground_truth_map.get(data_id, [])
        dlg_id = rec.get("dialogue_id") or dialogue_map.get(data_id)

        print(f"[{idx}/{total}] ID: {data_id}")

        # 出力ミス(例: TimeoutError)は別扱い
        if "error" in rec and rec["error"]:
            # 全体
            overall_error += 1

            # ツール使用/不使用カテゴリ別のエラー加算
            if gold:
                use_total += 1
                use_error += 1
            else:
                nouse_total += 1
                nouse_error += 1

            # 合算(使用・不使用判断)のエラー
            use_or_nouse_error += 1

            # ログ出力
            case_label = "ツール使用時" if gold else "ツール不使用時"
            print(f"• 出力ミス({case_label}): {rec['error']}")
            print("-" * 80)
            continue

        predicted = normalize_tool_calls(predicted_calls)
        expected = normalize_tool_calls(gold)

        # カテゴリ種別
        gold_used = bool(gold)
        pred_used = bool(predicted)

        # 全体(厳密一致)
        if predicted == expected:
            overall_correct += 1
            print("✓ 正解(全体: 厳密一致)")
        else:
            overall_incorrect += 1
            print("✗ 不正解(全体: 厳密不一致)")

        # ツール使用判断(ゴールドあり -> ツールを呼んだかのみ)
        if gold_used:
            use_total += 1
            if pred_used:
                use_correct += 1
            else:
                use_incorrect += 1
                incorrect_use_judgement.append({
                    "data_id": data_id,
                    "dialogue_id": dlg_id,
                    "Incorrect_genre": "ツール使用判断",
                    "question": question_map.get(data_id),
                    "output": rec.get("assistant") or rec.get("message") or rec.get("output") or rec.get("text") or rec.get("response") or rec.get("tool_calls"),
                    "ground_truth": gold,
                })
        else:
            # ツール不使用判断(ゴールドなし -> ツール未呼び出しを正)
            nouse_total += 1
            if not pred_used:
                nouse_correct += 1
            else:
                nouse_incorrect += 1
                incorrect_nouse_judgement.append({
                    "data_id": data_id,
                    "dialogue_id": dlg_id,
                    "Incorrect_genre": "ツール不使用判断",
                    "question": question_map.get(data_id),
                    "output": rec.get("assistant") or rec.get("message") or rec.get("output") or rec.get("text") or rec.get("response") or rec.get("tool_calls"),
                    "ground_truth": gold,
                })

        # 使用・不使用判断(合算): bool(gold) == bool(pred)
        if gold_used == pred_used:
            use_or_nouse_correct += 1
        else:
            use_or_nouse_incorrect += 1

        # tool call精度(ゴールドあり のみ厳密比較)
        if gold_used:
            call_total += 1
            if predicted == expected:
                call_correct += 1
            else:
                call_incorrect += 1
                incorrect_call_precision.append({
                    "data_id": data_id,
                    "dialogue_id": dlg_id,
                    "Incorrect_genre": "tool call精度",
                    "question": question_map.get(data_id),
                    "output": rec.get("assistant") or rec.get("message") or rec.get("output") or rec.get("text") or rec.get("response") or rec.get("tool_calls"),
                    "ground_truth": gold,
                })

        print("-" * 80)

    # 各種正答率
    overall_evaluated = total - overall_error
    overall_acc = (overall_correct / overall_evaluated * 100) if overall_evaluated else 0.0

    use_evaluated = use_total - use_error
    use_acc = (use_correct / use_evaluated * 100) if use_evaluated else 0.0

    nouse_evaluated = nouse_total - nouse_error
    nouse_acc = (nouse_correct / nouse_evaluated * 100) if nouse_evaluated else 0.0

    use_or_nouse_total = total
    use_or_nouse_evaluated = use_or_nouse_total - use_or_nouse_error
    use_or_nouse_acc = (use_or_nouse_correct / use_or_nouse_evaluated * 100) if use_or_nouse_evaluated else 0.0

    call_evaluated = call_total  # 本指標は出力ミスを対象外としている
    call_acc = (call_correct / call_evaluated * 100) if call_evaluated else 0.0

    m = re.match(r"result_(.+)\.json$", args.result.name)
    safe_model_name = m.group(1) if m else "unknown"
    output_path = Path(f"score_{safe_model_name}.json")

    with open(output_path, "w", encoding="utf-8") as out_f:
        # 1) 全体(厳密一致)
        all_summary = {
            "評価項目": "全体",
            "総データ数": total,
            "評価対象数": overall_evaluated,
            "正解数": overall_correct,
            "不正解数": overall_incorrect,
            "出力ミスのデータ数": overall_error,
            "全体の正答率(%)": round(overall_acc, 2),
        }

        # 2) ツール使用判断(ゴールドあり -> ツール呼び出し有無のみ)
        used_summary = {
            "評価項目": "ツール使用判断",
            "総データ数": use_total,
            "評価対象数": use_evaluated,
            "正解数": use_correct,
            "不正解数": use_incorrect,
            "出力ミスのデータ数": use_error,
            "全体の正答率(%)": round(use_acc, 2),
        }

        # 3) ツール不使用判断(ゴールドなし -> ツール未呼び出し)
        unused_summary = {
            "評価項目": "ツール不使用判断",
            "総データ数": nouse_total,
            "評価対象数": nouse_evaluated,
            "正解数": nouse_correct,
            "不正解数": nouse_incorrect,
            "出力ミスのデータ数": nouse_error,
            "全体の正答率(%)": round(nouse_acc, 2),
        }

        # 4) ツール使用・不使用判断(合算)
        use_or_nouse_summary = {
            "評価項目": "ツール使用・不使用判断",
            "総データ数": use_or_nouse_total,
            "評価対象数": use_or_nouse_evaluated,
            "正解数": use_or_nouse_correct,
            "不正解数": use_or_nouse_incorrect,
            "出力ミスのデータ数": use_or_nouse_error,
            "全体の正答率(%)": round(use_or_nouse_acc, 2),
        }

        # 5) tool call精度(ゴールドあり の厳密一致)
        call_summary = {
            "評価項目": "tool call精度",
            "総データ数": call_total,
            "評価対象数": call_evaluated,
            "正解数": call_correct,
            "不正解数": call_incorrect,
            "出力ミスのデータ数": 0,
            "全体の正答率(%)": round(call_acc, 2),
        }

        out_f.write(json.dumps(all_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(used_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(unused_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(use_or_nouse_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(call_summary, ensure_ascii=False) + "\n")

        # 誤答ログ: 指定順で出力
        for bad in incorrect_call_precision:
            out_f.write(json.dumps(bad, ensure_ascii=False) + "\n")
        for bad in incorrect_use_judgement:
            out_f.write(json.dumps(bad, ensure_ascii=False) + "\n")
        for bad in incorrect_nouse_judgement:
            out_f.write(json.dumps(bad, ensure_ascii=False) + "\n")

    # # 不正解データからランダム抽出してExcel向けCSVを出力
    # all_incorrect = (
    #     incorrect_call_precision + incorrect_use_judgement + incorrect_nouse_judgement
    # )
    # sample_size = min(100, len(all_incorrect))
    # sample_incorrect = (
    #     random.sample(all_incorrect, sample_size) if sample_size > 0 else []
    # )

    # # CSVはジャンル→id昇順で並べ替え
    # genre_order = {
    #     "tool call精度": 0,
    #     "ツール使用判断": 1,
    #     "ツール不使用判断": 2,
    # }

    # def _id_as_int(v):
    #     try:
    #         return int(v)
    #     except Exception:
    #         try:
    #             return int(str(v))
    #         except Exception:
    #             return float("inf")

    # def _csv_sort_key(item):
    #     g = item.get("Incorrect_genre") or ""
    #     gid = genre_order.get(g, 999)
    #     iid = item.get("data_id")
    #     return (gid, _id_as_int(iid), str(iid))

    # sample_incorrect_for_csv = sorted(sample_incorrect, key=_csv_sort_key)

    # csv_path = Path(f"sample_incorrect_{safe_model_name}.csv")

    # def _to_text(val):
    #     if isinstance(val, (dict, list)):
    #         try:
    #             return json.dumps(val, ensure_ascii=False)
    #         except Exception:
    #             return str(val)
    #     if isinstance(val, str):
    #         return val
    #     if val is None:
    #         return ""
    #     return str(val)

    # with open(csv_path, "w", encoding="utf-8-sig", newline="") as csvfile:
    #     writer = csv.DictWriter(
    #         csvfile,
    #         fieldnames=[
    #             "data_id",
    #             "dialogue_id",
    #             "Incorrect_genre",
    #             "question",
    #             "output",
    #             "ground_truth",
    #         ],
    #         delimiter=",",
    #         quoting=csv.QUOTE_ALL,
    #     )
    #     writer.writeheader()
    #     for item in sample_incorrect_for_csv:
    #         writer.writerow(
    #             {
    #                 "data_id": item.get("data_id") or item.get("id"),
    #                 "dialogue_id": item.get("dialogue_id"),
    #                 "Incorrect_genre": item.get("Incorrect_genre"),
    #                 "question": _to_text(item.get("question")),
    #                 "output": _to_text(item.get("output")),
    #                 "ground_truth": _to_text(item.get("ground_truth")),
    #             }
    #         )


    print(f"\n{'='*80}")
    print("評価結果(要約)")
    print(f"{'='*80}")
    print(f"全体: 総{total} / 対象{overall_evaluated} / ミス{overall_error} / 正答率{overall_acc:.2f}%")
    print(f"ツール使用判断: 総{use_total} / 対象{use_evaluated} / ミス{use_error} / 正答率{use_acc:.2f}%")
    print(f"ツール不使用判断: 総{nouse_total} / 対象{nouse_evaluated} / ミス{nouse_error} / 正答率{nouse_acc:.2f}%")
    print(f"ツール使用・不使用判断: 総{use_or_nouse_total} / 対象{use_or_nouse_evaluated} / ミス{use_or_nouse_error} / 正答率{use_or_nouse_acc:.2f}%")
    print(f"tool call精度: 総{call_total} / 対象{call_evaluated} / ミス0 / 正答率{call_acc:.2f}%")
    print(f"{'='*80}")
    print(f"評価結果のJSONを書き出しました: {output_path}")
    # print(
    #     f"不正解サンプル{sample_size}件のCSVを書き出しました: {csv_path}"
    # )


if __name__ == "__main__":
    main()
