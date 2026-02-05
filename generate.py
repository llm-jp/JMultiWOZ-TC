import json
from pathlib import Path
import argparse


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

    args = parser.parse_args()  # 引数を解析

    tools = load_tools(args.tools)  # ツールリストの読み込み
    input_data = load_jsonl(args.input)  # 入力データの読み込み
