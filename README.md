# JMultiWOZ-TC
マルチターン対話におけるエージェントの Function Calling 能力を、ツール使用有無の判断とツール呼び出し精度の両面から評価するためのリポジトリです。

論文サイト: https://cl-ait.github.io/Website-JMultiWOZ-TC/

## 概要
- サーバ: vLLM を OpenAI 互換エンドポイントとして起動（`--enable-auto-tool-choice` と `--tool-call-parser hermes` を使用）
- 生成: [generate.py](generate.py) が OpenAI SDK 経由でモデルに問い合わせ、NDJSON 形式の `result_{model}.json` を出力
- 評価: [evaluate.py](evaluate.py) が `tool_use_ja_ground.json` とモデル出力を比較し、主要指標のサマリと `score_{model}.json` を出力

必要ファイル（ルート直下）
- `tool_use_ja_input.json`: 入力（ユーザ質問/対話）NDJSON
- `tool_use_ja_ground.json`: 正解ツール呼び出し NDJSON

これらは大きい場合があります。未配置の場合は適切に取得・展開して配置してください（例: 配布物や [JMultiWOZ-TC_data.zip](JMultiWOZ-TC_data.zip) の展開）。

## 動作要件
- Python 3.10+
- GPU 環境推奨（vLLM のモデル推論のため）。CPU でも起動はできますが大規模モデルは非推奨です。

## セットアップ
```bash
# 仮想環境の作成と有効化（macOS/Linux）
python -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
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

生成が完了すると、モデル名に応じて `result_{safe_model_name}.json`（NDJSON）が作成されます。

3) 評価
```bash
python evaluate.py --result result_{safe_model_name}.json
```

評価結果のサマリがコンソールに表示され、詳細は `score_{safe_model_name}.json` に書き出されます。

## バッチ/クラスタ実行（任意）
Slurm での実行テンプレートとして [generate.sh](generate.sh) を用意しています。使用モデルとパーサを編集してから投入してください。
```bash
# 例: generate.sh 内の変数を編集
MODEL="Qwen/Qwen3-32B"
PORT=8000
parser="hermes"

# （環境に合わせて）sbatch で投入
sbatch generate.sh
```

スクリプトは vLLM サーバの起動→`generate.py` の実行→サーバ停止の手順を自動化します。GPU 台数やメモリ利用率などはスクリプト内の引数で調整してください。

## ファイル構成
- [generate.py](generate.py):
	- OpenAI 互換エンドポイント（既定: `http://localhost:8000/v1`）に接続
	- `tool_use_ja_input.json` を逐次処理し、Function Calling の結果を `result_{safe_model_name}.json` に追記
	- 途中再開に対応（既存出力の `data_id` をスキップ）
- [evaluate.py](evaluate.py):
	- `--result` で指定した NDJSON と `tool_use_ja_ground.json` を比較
	- 指標: 全体厳密一致、ツール使用判断、ツール不使用判断、両者合算、tool call 精度（厳密一致）
	- 出力: サマリ5行 + 誤答ログ（NDJSON）を `score_{safe_model_name}.json` に保存
- [generate.sh](generate.sh): Slurm 用ジョブスクリプト（任意）

## 動作しない場合
- vLLM が立ち上がらない/重い: モデルサイズに見合った GPU/メモリを確保し、`--tensor-parallel-size` や `--gpu-memory-utilization` を調整してください。
- 生成がタイムアウトする: `generate.py` はタイムアウト時に自動で数回リトライし、最終的に該当 `data_id` を `TimeoutError` として出力します。サーバ/モデル負荷を下げるか、ネットワーク状態を確認してください。
- モデル名が不正: vLLM でロード可能なモデル名を指定してください（例: `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`）。

## 参考: vLLM対応モデルとツールパーサ
- 対応モデル一覧（公式）: https://docs.vllm.ai/en/stable/models/supported_models.html
- Tool Calling（モデル別パーサまとめ・公式）: https://docs.vllm.ai/en/stable/features/tool_calling.html
	- Hermes 系: `--tool-call-parser hermes`
	- Mistral 系: `--tool-call-parser mistral`
	- Llama 3.1/3.2: `--tool-call-parser llama3_json`（JSONベース）
	- Llama 4: `--tool-call-parser llama4_pythonic`（Pythonic推奨）
	- Qwen 2.5 系: `--tool-call-parser hermes`（公式に記載あり）

詳細・対象モデルの最新情報は上記ドキュメントの各セクション（Hermes/Mistral/Llama/Qwen など）を参照してください。`--enable-auto-tool-choice` と併用することで自動ツール選択が有効になります。

## ライセンス/クレジット
データセットやモデルのライセンスに従ってご利用ください。本リポジトリの目的は評価手順の提供です。
