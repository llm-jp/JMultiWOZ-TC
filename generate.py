import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tools",
        default="tools.json",
        help="JMultiWOZ-TCに含まれるツールリストのファイルパスを指定",
    )
    parser.add_argument(
        "--input",
        default="jmultiwoz_tc_input.json",
        help="JMultiWOZ-TCに含まれる入力データのファイルパスを指定",
    )

    args = parser.parse_args()  # 引数を解析
