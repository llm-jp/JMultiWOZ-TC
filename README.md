# JMultiWOZ-TC
AIエージェントのツール呼び出しを評価するためのデータセットです。本データセットは，JMultiWOZを加工することで構築しており、4,246 対話に含まれるユーザ発話 31,303 発話に対して、合計 16,510 個のツール呼び出しが付与されています。

論文サイト: https://cl-ait.github.io/Website-JMultiWOZ-TC/

## AIエージェント用評価データセット
- 概要: AIエージェントのツール呼び出し精度を評価するための日本語データセットです。
- 由来: JMultiWOZ を再構築し、各ユーザ発話にツール呼び出しを付与しています。
- 想定用途: `jmultiwoz_tc_input.json` をモデル入力に用い、モデルが生成したツール呼び出しを `jmultiwoz_tc_ground.json` と比較して評価します。

### データ構成（展開後）
`JMultiWOZ-TC_data.zip` を解凍すると、以下のディレクトリが展開されます。

```
JMultiWOZ-TC_data/
├── jmultiwoz_tc_input.json   # 評価入力（ユーザ発話・コンテキスト）
└── jmultiwoz_tc_ground.json  # 正解ツール呼び出し（評価用アノテーション）
```

### 展開方法（macOS）
- Finderで `JMultiWOZ-TC_data.zip` をダブルクリック
- またはターミナル:

```bash
unzip JMultiWOZ-TC_data.zip
```

### ファイル説明
- `jmultiwoz_tc_input.json`: モデルがツール呼び出しを推定するための入力（ユーザ発話や対話コンテキスト）。
- `jmultiwoz_tc_ground.json`: 対応する正解のツール呼び出しアノテーション（評価指標算出に使用）。

## スクリプト
- 公開準備中

## ライセンス/クレジット
JMultiWOZ-TC データは Creative Commons Attribution 4.0 International (CC BY 4.0) で公開します。

詳細: https://creativecommons.org/licenses/by/4.0/

## 謝辞
JMultiWOZ-TCは、JMultiWOZ に基づく対話データをツール呼び出し形式へ再構築した評価用データセットです。

JMultiWOZ GitHub: https://github.com/nu-dialogue/jmultiwoz