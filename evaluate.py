import json
import argparse
import re
from pathlib import Path
from generate import load_jsonl


def canonicalize_arguments(args: dict):
    """ツール引数を正規化した文字列へ変換

    generate.py(vLLM 経由)の出力および JMultiWOZ-TC の ground_truth に含まれる
    arguments フィールド（dict）を、比較・保存に適した一貫した
    文字列表現へ変換する。

    引数は dict であることを前提とし、キー順をソートして JSON 文字列に変換する。

    例: 入力: {"name": "abcdef", "area": "北区"}
        出力: '{"area": "北区", "name": "abcdef"}'

    Args:
        args (dict): ツール呼び出しの引数。

    Returns:
        str: 正規化された引数の文字列表現。
    """
    return json.dumps(args, sort_keys=True, ensure_ascii=False)



def normalize_tool_calls(tool_calls):
    """ツール呼び出しを比較可能な集合へ正規化

    ツール呼び出し配列を (関数名, 正規化した引数文字列) の集合へ変換する。

    Args:
        tool_calls (list[dict]): ツール呼び出し配列。

    Returns:
        set: `(name, canonical_args_str)` の集合。
    """
    if not tool_calls:
        return set()

    normalized = []
    for call in tool_calls:
        if not isinstance(call, dict):
            # 想定外フォーマットは無視する
            continue

        name = call.get("name")
        raw_args = call.get("arguments")

        if name is None:
            # name が無いものは比較対象外
            continue

        args_canonical = canonicalize_arguments(raw_args)
        normalized.append((name, args_canonical))

    return set(normalized)



def build_question_and_dialogue_maps(input_path: Path):
    """質問・ダイアログIDのマップ作成

    JMultiWOZ-TC の入力データから `data_id` をキーとして、
    `question` と `dialogue_id` のマップを作成する。

    Args:
        input_path (Path): JMultiWOZ-TC の入力データファイルのパス。

    Returns:
        tuple: `(data_id2question, data_id2dialogue_id)` の2要素タプル。
            - data_id2question (dict): `data_id` をキー、`question` を値とする辞書。
            - data_id2dialogue_id (dict): `data_id` をキー、`dialogue_id` を値とする辞書。
    """
    data_id2question = {}
    data_id2dialogue_id = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            qid = obj.get("data_id")
            if qid is not None:
                data_id2question[qid] = obj.get("question")
                data_id2dialogue_id[qid] = obj.get("dialogue_id")
    return data_id2question, data_id2dialogue_id



def build_ground_truth_map(ground_items: list):
    """正解データのマップ作成

    JMultiWOZ-TC の正解データから、`data_id` をキーに `ground_truth` を引ける
    辞書を作成する。

    Args:
        ground_items (list): 正解データのレコード配列。

    Returns:
        dict: `data_id` をキー、`ground_truth`(list) を値とする辞書。
    """
    data_id2ground_truth = {}
    for item in ground_items:
        key = item.get("data_id")
        data_id2ground_truth[key] = item.get("ground_truth", [])
    return data_id2ground_truth



def aggregate_overall_metrics(
    result_data,
    data_id2ground_truth,
    data_id2question=None,
    data_id2dialogue_id=None,
    incorrect_genre=None,
):
    """全体評価の集計

    全レコードを走査し、モデル出力の tool call 集合と正解の tool call 集合が
    厳密一致しているかを判定して、全体指標を集計する。
    `error` が入っているレコードは評価対象外として `error` に加算する。

    Args:
        result_data (list): LLM出力のレコード配列。
        data_id2ground_truth (dict): `data_id` をキーにした `ground_truth` を値とする辞書。
        data_id2question (dict): `data_id` から質問文を引く辞書。
        data_id2dialogue_id (dict): `data_id` から対話IDを引く辞書。
        incorrect_genre (str): 不正解ログに入れるジャンル名。

    Returns:
        tuple: `(stats, incorrect_logs)` の2要素タプル。
            - stats (dict): 全体評価の集計結果。
                - total (int): 総データ件数。
                - evaluated (int): 評価対象件数 (total - error)。
                - correct (int): 厳密一致した件数。
                - incorrect (int): 厳密不一致の件数。
                - error (int): 出力ミス等により未評価となった件数。
                - acc (float): 正答率(%)。
            - incorrect_logs (list): 不正解ケースのログ配列。
    """
    total = len(result_data)
    correct = 0
    incorrect = 0
    error = 0
    incorrect_logs = []

    for rec in result_data:
        data_id = rec.get("data_id")
        output_calls = rec.get("tool_calls", [])
        ground_truth_calls = data_id2ground_truth.get(data_id, [])

        if rec.get("error"):
            error += 1
            continue

        output_calls_set = normalize_tool_calls(output_calls)
        ground_truth_set = normalize_tool_calls(ground_truth_calls)

        if output_calls_set == ground_truth_set:
            correct += 1
        else:
            incorrect += 1
            dlg_id = rec.get("dialogue_id") or (
                data_id2dialogue_id.get(data_id) if data_id2dialogue_id else None
            )
            incorrect_logs.append(
                {
                    "data_id": data_id,
                    "dialogue_id": dlg_id,
                    "Incorrect_genre": incorrect_genre,
                    "question": data_id2question.get(data_id) if data_id2question else None,
                    "output": output_calls,
                    "ground_truth": ground_truth_calls,
                }
            )

    evaluated = correct + incorrect
    acc = correct / evaluated * 100 if evaluated > 0 else 0

    stats = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "error": error,
        "acc": acc,
    }


    return stats, incorrect_logs



def aggregate_tool_usage_metrics(result_data, data_id2ground_truth, data_id2question, data_id2dialogue_id,):
    """ツール使用/不使用判断の集計

    全レコードを走査し、正解がツール使用か不使用かで分けて評価する。
    - 正解がツール使用ケース (`ground_truth` が非空)
    - 正解がツール不使用ケース (`ground_truth` が空)

    不正解ケースは用途別にログへ保存する。
    `error` レコードは、正解側の使用/不使用に応じて `error` のみ加算する。

    Args:
        result_data (list): LLM出力のレコード配列。
        data_id2ground_truth (dict): `data_id` をキーにした `ground_truth` を値とする辞書。
        data_id2question (dict): `data_id` をキーにした `question` を値とする辞書。
        data_id2dialogue_id (dict): `data_id` をキーにした `dialogue_id` を値とする辞書。

    Returns:
        tuple: `(use_stats, nouse_stats, incorrect_use_judgement, incorrect_nouse_judgement)`
            の4要素タプル。
            - use_stats (dict): ツール使用判断の集計結果
              (`total`, `evaluated`, `correct`, `incorrect`, `error`, `acc`)。
            - nouse_stats (dict): ツール不使用判断の集計結果
              (`total`, `evaluated`, `correct`, `incorrect`, `error`, `acc`)。
            - incorrect_use_judgement (list):
              本来ツールを使用するケースの誤答ログ配列。
            - incorrect_nouse_judgement (list):
              本来ツールを使用しないケースの誤答ログ配列。
    """
    use_total = 0
    use_error = 0
    use_correct = 0
    use_incorrect = 0

    nouse_total = 0
    nouse_error = 0
    nouse_correct = 0
    nouse_incorrect = 0

    incorrect_use_judgement = []
    incorrect_nouse_judgement = []

    for rec in result_data:
        data_id = rec.get("data_id")
        output_calls = rec.get("tool_calls", [])
        ground_truth_calls = data_id2ground_truth.get(data_id, [])
        dlg_id = rec.get("dialogue_id") or data_id2dialogue_id.get(data_id)

        if rec.get("error"):
            if ground_truth_calls:
                use_error += 1
            else:
                nouse_error += 1
            continue

        output_calls_set = normalize_tool_calls(output_calls)
        ground_truth_set = normalize_tool_calls(ground_truth_calls)

        judgement_correct = len(output_calls_set) > 0 and len(ground_truth_set) > 0

        if ground_truth_set:
            if judgement_correct:
                use_correct += 1
            else:
                use_incorrect += 1
                incorrect_use_judgement.append(
                    {
                        "data_id": data_id,
                        "dialogue_id": dlg_id,
                        "incorrect_genre": "ツール使用判断",
                        "question": data_id2question.get(data_id),
                        "output": output_calls,
                        "ground_truth": ground_truth_calls,
                    }
                )
        else:
            if judgement_correct:
                nouse_correct += 1
            else:
                nouse_incorrect += 1
                incorrect_nouse_judgement.append(
                    {
                        "data_id": data_id,
                        "dialogue_id": dlg_id,
                        "incorrect_genre": "ツール不使用判断",
                        "question": data_id2question.get(data_id),
                        "output": output_calls,
                        "ground_truth": ground_truth_calls,
                    }
                )

    use_total = use_correct + use_incorrect + use_error
    use_evaluated = use_total - use_error
    use_acc = use_correct / use_evaluated * 100 if use_evaluated > 0 else 0

    nouse_total = nouse_correct + nouse_incorrect + nouse_error
    nouse_evaluated = nouse_total - nouse_error
    nouse_acc = nouse_correct / nouse_evaluated * 100 if nouse_evaluated > 0 else 0

    use_stats = {
        "total": use_total,
        "evaluated": use_evaluated,
        "correct": use_correct,
        "incorrect": use_incorrect,
        "error": use_error,
        "acc": use_acc,
    }
    nouse_stats = {
        "total": nouse_total,
        "evaluated": nouse_evaluated,
        "correct": nouse_correct,
        "incorrect": nouse_incorrect,
        "error": nouse_error,
        "acc": nouse_acc,
    }

    return (
        use_stats,
        nouse_stats,
        incorrect_use_judgement,
        incorrect_nouse_judgement,
    )


def aggregate_tool_call_metrics(
    result_data,
    data_id2ground_truth,
    data_id2question,
    data_id2dialogue_id,
):
    """tool call精度の集計

    正解がツールを使用するケース (`ground_truth` が非空) に限定して
    tool call 内容の厳密一致精度を集計する。
    `aggregate_overall_metrics` を再利用し、正解がツールを使用するケースのみをフィルタして評価する。
    不一致ケースは誤答ログに保存する。
    対象ケースで `error` がある場合は `error` のみ加算して未評価とする。

    Args:
        result_data (list): LLM出力のレコード配列。
        data_id2ground_truth (dict): `data_id` をキーにした `ground_truth` を値とする辞書。
        data_id2question (dict): `data_id` をキーにした `question` を値とする辞書。
        data_id2dialogue_id (dict): `data_id` をキーにした `dialogue_id` を値とする辞書。

    Returns:
        tuple: `(func_stats, incorrect_call_precision)` の2要素タプル。
            - func_stats (dict): tool call 精度の集計結果
              (`total`, `evaluated`, `correct`, `incorrect`, `error`, `acc`)。
            - incorrect_call_precision (list):
              tool call 精度が不一致だったケースの誤答ログ配列。
    """
    filtered_data = [
        rec for rec in result_data
        if data_id2ground_truth.get(rec.get("data_id"), [])
    ]

    func_stats, incorrect_call_precision = aggregate_overall_metrics(
        filtered_data,
        data_id2ground_truth,
        data_id2question=data_id2question,
        data_id2dialogue_id=data_id2dialogue_id,
        incorrect_genre="tool call精度",
    )

    return func_stats, incorrect_call_precision


def evaluate_results(result_data, data_id2ground_truth, data_id2question, data_id2dialogue_id):
    """結果比較とメトリクス集計

    LLM出力と正解データを比較し、各種指標(全体/使用判断/不使用判断/合算/tool call精度)を集計して返す。
    出力ミス(TimeoutError など)は別扱いでカウントする。
    同時に正答率も計算して返す。

    Args:
        result_data (list): LLM出力のレコード配列。
        data_id2ground_truth (dict): `data_id` をキーにした `ground_truth` を値とする辞書。
        data_id2question (dict): `data_id` をキーにした `question` を値とする辞書。
        data_id2dialogue_id (dict): `data_id` をキーにした `dialogue_id` を値とする辞書。

    Returns:
        tuple: `(accuracies, incorrect_call_precision, incorrect_use_judgement, incorrect_nouse_judgement)` の4要素のタプル。
            - accuracies (dict): 各種指標の正答率を含む要約を格納した辞書。
            - incorrect_call_precision (list): ツール呼び出し精度が誤っているケースのログ。
            - incorrect_use_judgement (list): ツール使用判断が誤っているケースのログ。
            - incorrect_nouse_judgement (list): ツール不使用判断が誤っているケースのログ。
    """
    total = len(result_data)
    print(f"評価開始: {total}件のデータを処理します\n")

    overall_stats, _ = aggregate_overall_metrics(result_data, data_id2ground_truth)

    (use_stats, nouse_stats, incorrect_use_judgement, incorrect_nouse_judgement,) = aggregate_tool_usage_metrics(result_data, data_id2ground_truth, data_id2question, data_id2dialogue_id,)

    # use_stats と nouse_stats を統合して use_or_nouse_stats を計算
    use_or_nouse_total = use_stats["total"] + nouse_stats["total"]
    use_or_nouse_error = use_stats["error"] + nouse_stats["error"]
    use_or_nouse_correct = use_stats["correct"] + nouse_stats["correct"]
    use_or_nouse_incorrect = use_stats["incorrect"] + nouse_stats["incorrect"]
    use_or_nouse_evaluated = use_or_nouse_total - use_or_nouse_error
    use_or_nouse_acc = use_or_nouse_correct / use_or_nouse_evaluated * 100 if use_or_nouse_evaluated > 0 else 0

    use_or_nouse_stats = {
        "total": use_or_nouse_total,
        "evaluated": use_or_nouse_evaluated,
        "correct": use_or_nouse_correct,
        "incorrect": use_or_nouse_incorrect,
        "error": use_or_nouse_error,
        "acc": use_or_nouse_acc,
    }

    call_stats, incorrect_call_precision = aggregate_tool_call_metrics(
        result_data,
        data_id2ground_truth,
        data_id2question,
        data_id2dialogue_id,
    )

    accuracies = {
        "overall": overall_stats,
        "used": use_stats,
        "unused": nouse_stats,
        "use_or_nouse": use_or_nouse_stats,
        "call": call_stats,
    }

    return (
        accuracies,
        incorrect_call_precision,
        incorrect_use_judgement,
        incorrect_nouse_judgement,
    )



def write_summary(
    output_path: Path,
    accuracies: dict,
):
    """要約の書き出し

    計算済みの要約(5行)をJSONL形式で出力する。
    出力の順序は次の通り:
        - 全体の要約 (1行)
        - ツール使用判断の要約 (1行)
        - ツール不使用判断の要約 (1行)
        - ツール使用・不使用判断の要約 (1行)
        - tool call精度の要約 (1行)

    Args:
        output_path (Path): 出力ファイルパス。
        accuracies (dict): 各種指標の要約を格納した辞書。

    Returns:
        None: なし。
    """
    overall = accuracies["overall"]
    used = accuracies["used"]
    unused = accuracies["unused"]
    use_or_nouse = accuracies["use_or_nouse"]
    call = accuracies["call"]

    with open(output_path, "w", encoding="utf-8") as out_f:
        all_summary = {
            "評価項目": "全体",
            "総データ数": overall["total"],
            "評価対象数": overall["evaluated"],
            "正解数": overall["correct"],
            "不正解数": overall["incorrect"],
            "出力ミスのデータ数": overall["error"],
            "全体の正答率(%)": overall["acc"],
        }
        used_summary = {
            "評価項目": "ツール使用判断",
            "総データ数": used["total"],
            "評価対象数": used["evaluated"],
            "正解数": used["correct"],
            "不正解数": used["incorrect"],
            "出力ミスのデータ数": used["error"],
            "全体の正答率(%)": used["acc"],
        }
        unused_summary = {
            "評価項目": "ツール不使用判断",
            "総データ数": unused["total"],
            "評価対象数": unused["evaluated"],
            "正解数": unused["correct"],
            "不正解数": unused["incorrect"],
            "出力ミスのデータ数": unused["error"],
            "全体の正答率(%)": unused["acc"],
        }
        use_or_nouse_summary = {
            "評価項目": "ツール使用・不使用判断",
            "総データ数": use_or_nouse["total"],
            "評価対象数": use_or_nouse["evaluated"],
            "正解数": use_or_nouse["correct"],
            "不正解数": use_or_nouse["incorrect"],
            "出力ミスのデータ数": use_or_nouse["error"],
            "全体の正答率(%)": use_or_nouse["acc"],
        }
        call_summary = {
            "評価項目": "tool call精度",
            "総データ数": call["total"],
            "評価対象数": call["evaluated"],
            "正解数": call["correct"],
            "不正解数": call["incorrect"],
            "出力ミスのデータ数": call["error"],
            "全体の正答率(%)": call["acc"],
        }
        
        out_f.write(json.dumps(all_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(used_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(unused_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(use_or_nouse_summary, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(call_summary, ensure_ascii=False) + "\n")


def write_incorrect_logs(
    output_path: Path,
    incorrect_call_precision: list,
    incorrect_use_judgement: list,
    incorrect_nouse_judgement: list,
):
    """誤答ログの書き出し

    誤答ログをJSONL形式で出力する。
    出力の順序は次の通り:
        - tool call精度の誤答ログをすべて
        - ツール使用判断の誤答ログをすべて
        - ツール不使用判断の誤答ログをすべて

    Args:
        output_path (Path): 出力ファイルパス。
        incorrect_call_precision (list):
            ツール呼び出し内容そのものが誤っているケースのログ
            ("tool call精度" の誤答)。
        incorrect_use_judgement (list):
            本来ツールを使うべきなのに使わなかったケースのログ
            (「ツール使用判断」の誤答)。
        incorrect_nouse_judgement (list):
            本来ツールを使うべきでないのに使ったケースのログ
            (「ツール不使用判断」の誤答)。

    Returns:
        None: なし。
    """
    with open(output_path, "w", encoding="utf-8") as out_f:
        for bad in incorrect_call_precision:
            out_f.write(json.dumps(bad, ensure_ascii=False) + "\n")
        for bad in incorrect_use_judgement:
            out_f.write(json.dumps(bad, ensure_ascii=False) + "\n")
        for bad in incorrect_nouse_judgement:
            out_f.write(json.dumps(bad, ensure_ascii=False) + "\n")



def print_summary_to_console(accuracies: dict):
    """要約をコンソール出力

    要約結果を整形して標準出力に表示する。
    以下の5行をこの順番で出力する:
        - 全体の要約 (1行)
        - ツール使用判断の要約 (1行)
        - ツール不使用判断の要約 (1行)
        - ツール使用・不使用判断の要約 (1行)
        - tool call精度の要約 (1行)

    Args:
        accuracies (dict): 各種指標の要約を格納した辞書。

    Returns:
        None: なし。
    """
    overall = accuracies["overall"]
    used = accuracies["used"]
    unused = accuracies["unused"]
    use_or_nouse = accuracies["use_or_nouse"]
    call = accuracies["call"]

    print(f"\n{'='*80}")
    print("評価結果(要約)")
    print(f"{'='*80}")
    print(f"全体: 総{overall['total']} / 対象{overall['evaluated']} / ミス{overall['error']} / 正答率{overall['acc']:.2f}%")
    print(f"ツール使用判断: 総{used['total']} / 対象{used['evaluated']} / ミス{used['error']} / 正答率{used['acc']:.2f}%")
    print(f"ツール不使用判断: 総{unused['total']} / 対象{unused['evaluated']} / ミス{unused['error']} / 正答率{unused['acc']:.2f}%")
    print(f"ツール使用・不使用判断: 総{use_or_nouse['total']} / 対象{use_or_nouse['evaluated']} / ミス{use_or_nouse['error']} / 正答率{use_or_nouse['acc']:.2f}%")
    print(f"tool call精度: 総{call['total']} / 対象{call['evaluated']} / ミス{call['error']} / 正答率{call['acc']:.2f}%")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="LLM出力ファイル(result_{safe_model_name}.jsonl)のパスを指定",
    )
    parser.add_argument(
        "--ground",
        type=Path,
        default="jmultiwoz_tc_ground.jsonl",
        help="JMultiWOZ-TCに含まれる正解データのファイルパスを指定",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="jmultiwoz_tc_input.jsonl",
        help="JMultiWOZ-TCに含まれる入力データのファイルパスを指定",
    )

    args = parser.parse_args()

    result_data = load_jsonl(args.result)
    ground_data = load_jsonl(args.ground)

    data_id2question, data_id2dialogue_id = build_question_and_dialogue_maps(args.input)
    data_id2ground_truth = build_ground_truth_map(ground_data)

    accuracies, log_call, log_use, log_nouse = evaluate_results(
        result_data, data_id2ground_truth, data_id2question, data_id2dialogue_id
    )

    m = re.match(r"result_(.+)\.jsonl$", args.result.name)
    safe_model_name = m.group(1) if m else "unknown"
    summary_path = Path(f"score_{safe_model_name}.json")
    incorrect_path = Path(f"incorrect_{safe_model_name}.json")

    write_summary(summary_path, accuracies)
    write_incorrect_logs(incorrect_path, log_call, log_use, log_nouse)
    print_summary_to_console(accuracies)
    print(f"{'='*80}")
    print(f"評価結果(サマリー)を書き出しました: {summary_path}")
    print(f"評価結果(誤答ログ)を書き出しました: {incorrect_path}")


if __name__ == "__main__":
    main()
