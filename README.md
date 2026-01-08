# agent_Tool_use
マルチターン対話でのエージェントのfunction calling評価

# 論文サイト
https://cl-ait.github.io/Website-JMultiWOZ-TC/

# set up
以下のライブラリをインストール
```bash
python -m venv .venv
pip install vllm
source .venv/bin/activate
```

# 実行方法

Qwen3-14Bを評価に使用する場合
```bash
vllm serve Qwen/Qwen3-14B --port 8000 --enable-auto-tool-choice --tool-call-parser hermes
```

別のターミナルで実行
```bash
source .venv/bin/activate
python test.py
```


ジョブファイルを投げる場合は`generate.sh`の以下の部分を使用するモデルによって書き換えて実行する
```bash
MODEL="Qwen/Qwen3-32B"
PORT=8000
parser="hermes"
```

# 評価

`{MODEL_GENERATE_RESULT}.json`には`generate.py`によって生成されたjsonファイル名を入力する
`evaluate.py`を用いてモデルの出力と正解データを比較する
```bash
python evaluate.py --result {MODEL_GENERATE_RESULT}.json
```
