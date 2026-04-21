# JMultiWOZ-TC
AIエージェントのツール呼び出しを評価するためのデータセットです。本データセットは，JMultiWOZを加工することで構築しており、4,246 対話に含まれるユーザ発話 31,303 発話に対して、合計 16,510 個のツール呼び出しが付与されています。

## AIエージェント用評価データセット
`jmultiwoz_tc_input.jsonl` をモデル入力に用い、モデルが生成したツール呼び出しを `jmultiwoz_tc_ground.jsonl` と比較して評価します。

### データ構成

```bash
JMultiWOZ-TC/
├── generate.py            
├── evaluate.py 
└── dataset.zip            # ツール定義とデータセット（展開が必要）
    ├── tools.json             
    ├── jmultiwoz_tc_input.jsonl 
    └── jmultiwoz_tc_ground.jsonl 
```

## 概要
- サーバ: vLLM を OpenAI 互換エンドポイントとして起動（`--enable-auto-tool-choice` と `--tool-call-parser hermes` を使用）
- 生成: [generate.py](generate.py) が OpenAI SDK 経由でモデルに問い合わせ、JSONL 形式の `result_{safe_model_name}.jsonl` を出力
- 評価: [evaluate.py](evaluate.py) は `jmultiwoz_tc_ground.jsonl` とモデル出力を比較し、サマリを `score_{safe_model_name}.json`、誤答ログを `incorrect_{safe_model_name}.json` に出力する設計

### ファイル説明
- `tools.json`: ツール定義（Function Calling の仕様）。[generate.py](generate.py)が読み込みます。
- `jmultiwoz_tc_input.jsonl`: モデルがツール呼び出しを推定するための入力（ユーザ発話や対話コンテキスト）。
- `jmultiwoz_tc_ground.jsonl`: 対応する正解のツール呼び出しアノテーション（評価指標算出に使用）。

## 動作要件
- Python 3.10以上
- GPU 環境推奨（vLLM のモデル推論のため）。

## セットアップ
以下を順に実施してください。

1. データセットの展開
```bash
# dataset.zip を解凍（tools.json と各 JSONL ファイルが抽出されます）
unzip dataset.zip
```

2. Python環境の準備と依存インストール
```bash
# 仮想環境の作成と有効化（macOS/Linux）
python -m venv .venv
source .venv/bin/activate

# 依存関係（サーバ+クライアント）
# - vLLM: OpenAI互換APIサーバ
# - openai/httpx: クライアント（generate.py で使用）
pip install -U vllm openai httpx
```

## 使い方（ローカル）
1) vLLM サーバを起動（例: Qwen3-14B）
```bash
vllm serve Qwen/Qwen3-14B \
	--port 8000 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes
```

2) 生成実行（別ターミナル）
```bash
source .venv/bin/activate
python generate.py
```

生成が完了すると、モデル名に応じて `result_{safe_model_name}.jsonl` が作成されます。

3) 評価
```bash
python evaluate.py --result result_{safe_model_name}.jsonl
```

評価結果のサマリがコンソールに表示され、詳細は `score_{safe_model_name}.json` と `incorrect_{safe_model_name}.json` に書き出されます。

## 実行オプション

### generate.py
基本実行:
```bash
python generate.py
```

指定可能なオプション:
- `--tools` (既定: `tools.json`)
	- ツール定義ファイルのパス
- `--input` (既定: `jmultiwoz_tc_input.jsonl`)
	- 入力JSONLファイルのパス
- `--base-url` (既定: `http://localhost:8000/v1`)
	- vLLM(OpenAI互換API)のベースURL
- `--openai-api-key` (既定: `dummy`)
	- OpenAIクライアント初期化用のAPIキー
- `--output-dir` (既定: `.`)
	- `result_{safe_model_name}.jsonl` の出力先ディレクトリ
- `--max-retries` (既定: `3`)
	- タイムアウト時の再試行回数

実行例:
```bash
python generate.py \
	--base-url http://localhost:8000/v1 \
	--output-dir .qwen3-14b \
	--max-retries 5
```

### evaluate.py
基本実行:
```bash
python evaluate.py --result result_{safe_model_name}.jsonl
```

指定可能なオプション:
- `--result` (**必須**)
	- 評価対象のモデル出力JSONLファイル
- `--ground` (既定: `jmultiwoz_tc_ground.jsonl`)
	- 正解データJSONLファイル
- `--input` (既定: `jmultiwoz_tc_input.jsonl`)
	- 入力データJSONLファイル（誤答ログの補助情報参照に使用）

実行例:
```bash
python evaluate.py \
	--result result_Qwen_Qwen3-14B.jsonl \
	--ground jmultiwoz_tc_ground.jsonl \
	--input jmultiwoz_tc_input.jsonl
```

## 出力ファイル

### 生成結果: `result_{safe_model_name}.jsonl`
- 作成元: `python generate.py`
- 形式: JSONL
- 用途: 各 `data_id` に対するモデルのツール呼び出し出力を保存

主なレコード形式（成功時）:
```json
{
	"data_id": "...",
	"dialogue_id": "...",
	"tool_calls": [
		{"name": "tool_name", "arguments": {"key": "value"}}
	]
}
```

主なレコード形式（エラー発生時）:
```json
{
	"data_id": "...",
	"dialogue_id": "...",
	"tool_calls": [],
	"error": "TimeoutError"
}
```

### 評価サマリ: `score_{safe_model_name}.json`
- 作成元: `python evaluate.py --result ...`
- 形式: JSONL
- 行数: 5行固定
	- 全体
	- ツール使用判断
	- ツール不使用判断
	- ツール使用・不使用判断
	- tool call精度
- 各行の主キー:
	- `評価項目`, `総データ数`, `評価対象数`, `正解数`, `不正解数`, `出力ミスのデータ数`, `全体の正答率(%)`

#### 評価尺度

`全体の正答率(%)` は、いずれの行も次式で計算します。

```text
全体の正答率(%) = 正解数 / 評価対象数 * 100
評価対象数 = 正解数 + 不正解数
```

`出力ミスのデータ数` は、モデル出力に `error`（例: `TimeoutError`）が入った件数で、正誤判定の分母から除外されます。

**全体**
- 対象: 全データ
- 判定: 1件ごとに、モデルの `tool_calls` と正解 `ground_truth` が **完全一致** かどうか
- 何を見る指標か: 「ツールを使うかどうか」と「使う場合の呼び出し内容」の両方をまとめた最終精度

**ツール使用判断**
- 対象: 正解側で `ground_truth` が空でないデータ（本来ツールを使うべきケース）
- 判定: モデルが `tool_calls` を1件以上出せたか（使うべき場面で使えたか）
- 何を見る指標か: 必要な場面でツール使用を選択できるか

**ツール不使用判断**
- 対象: 正解側で `ground_truth` が空のデータ（本来ツール不要のケース）
- 判定: モデルが `tool_calls` を出さなかったか（不要な場面で呼ばないか）
- 何を見る指標か: 不要なツール呼び出しを抑制できるか

**ツール使用・不使用判断**
- 対象: 上記の**ツール使用判断**と**ツール不使用判断**を合わせた全データ
- 判定: 「使う/使わない」の二値判断が正しいか
- 何を見る指標か: ツール利用有無の判断能力（内容の正しさはここでは見ない）

**tool call精度**
- 対象: 正解側で `ground_truth` が空でないデータ（本来ツールを使うべきケース）
- 判定: `tool_calls` の内容が正解と **完全一致** か
  - 一致条件: 各 call の `name` と `arguments` が一致
  - `arguments` はキー順を正規化して比較するため、JSONキーの順序差は不問
  - call の比較は集合ベースのため、順序差は不問
- 何を見る指標か: 実際のツール呼び出し内容（関数名・引数）をどれだけ正確に出せるか

### 誤答ログ: `incorrect_{safe_model_name}.json`
- 作成元: `python evaluate.py --result ...`
- 形式: JSONL
- 用途: 誤答ケースを1行1JSONで保存
- 出力順:
	1. tool call精度の誤答
	2. ツール使用判断の誤答
	3. ツール不使用判断の誤答

## 動作しない場合
- vLLM が立ち上がらない/重い: モデルサイズに見合った GPU/メモリを確保し、`--tensor-parallel-size` や `--gpu-memory-utilization` を調整してください。
- 生成がタイムアウトする: `generate.py` はタイムアウト時に自動で数回リトライし、最終的に該当 `data_id` を `TimeoutError` として出力します。サーバ/モデル負荷を下げるか、ネットワーク状態を確認してください。

## ライセンス/クレジット
JMultiWOZ-TC データは Creative Commons Attribution 4.0 International (CC BY 4.0) で公開します。

詳細: https://creativecommons.org/licenses/by/4.0/

## 謝辞
JMultiWOZ-TCは、JMultiWOZ に基づく対話データをツール呼び出し形式へ再構築した評価用データセットです。

JMultiWOZ GitHub: https://github.com/nu-dialogue/jmultiwoz