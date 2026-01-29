import json
from pathlib import Path
import argparse


def load_jsonl(file_path):  # JSONファイルを読み込む
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_ndjsonl(file_path):  # NDJSONファイルを読み込む
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


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
        default="jmultiwoz_tc_input.json",
        help="JMultiWOZ-TCに含まれる入力データのファイルパスを指定",
    )

    args = parser.parse_args()  # 引数を解析

    tools = load_jsonl(args.tools)  # ツールリストの読み込み
    input_data = load_ndjsonl(args.input)  # 入力データの読み込み
