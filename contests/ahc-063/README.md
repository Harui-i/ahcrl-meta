# AHC063: Colorful Ouroboros

優勝しよう。

コンテスト URL: <https://atcoder.jp/contests/ahc063>

問題文 URL: <https://atcoder.jp/contests/ahc063/tasks/ahc063_a>

問題文: [problem_ja.md](problem_ja.md) / [problem_en.md](problem_en.md)

AtCoder提供のツール: `tools/`

## PPO訓練

```bash
uv run python3 -m ahcrl.contests.ahc063.train_ppo \
  --config contests/ahc-063/configs/ppo_train.toml
```

学習成果物は `contests/ahc-063/artifacts/ppo/run_*/` に保存される。各 run の
`checkpoint_latest.pt` が export 対象で、`config.json` からモデル構成と
観測正規化の状態を復元する。

## 学習済みモデルの export と評価

以下では、評価したい run を `RUN_DIR` に指定する。`--run-dir` を省略すると、
`artifacts/ppo` 以下で最も新しい、`checkpoint_latest.pt` を持つ run が使われる。

```bash
RUN_DIR=contests/ahc-063/artifacts/ppo/run_YYYYMMDD_HHMMSS

# run root の checkpoint_latest.pt を自己完結した C++ 提出コードへ変換する。
uv run python3 contests/ahc-063/scripts/export_torchscript_submit.py \
  --run-dir "$RUN_DIR"

# Pahcer がコンパイルする入力ファイルへ配置する。
cp "$RUN_DIR/submit.cpp" contests/ahc-063/eval/ahc063/main.cpp
```

exporter は TorchScript モデル、観測正規化、および盤面遷移を `submit.cpp` に埋め込む。
デフォルトは各手で最大 logit の合法手を選ぶ決定的な方策である。確率的に行動をサンプルしたい場合は
export 時に `--softmax` を追加する。

```bash
cd contests/ahc-063/eval/ahc063
pahcer run -c "run_YYYYMMDD_HHMMSS argmax"
```

`pahcer_config.toml` は seed 0〜99 を実行し、公式 visualizer でスコアを計算する。
生成物は `pahcer/` 以下に保存される。コンパイルには `uv sync` 済みの `.venv` に含まれる
libtorch を参照するため、依存関係をまだ用意していなければリポジトリのルートで先に実行する。

```bash
uv sync
```
