import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", default="tools.json")  # ツールリストの引数を追加
    parser.add_argument("--input", default="jmultiwoz_tc_input.json")
    # 入力データの引数を追加

    args = parser.parse_args()  # 引数を解析
