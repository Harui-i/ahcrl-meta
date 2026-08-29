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

環境は各時点までの公式絶対スコアの最小値を trajectory best として保持する。
`score` は trajectory best、reward は trajectory best を更新した量を `10000` で
割った非負値である。盤面上の現在スコアが best より悪い場合、その差も観測 plane に
含める。episode と提出コードの探索上限はともに `max_steps_per_cell * N^2` で、標準の
`max_steps_per_cell = 4` では N=8 が256手、N=16が1024手となる。

visualizer と提出コードが出力するのは best prefix ではなく、実際に選択した全行動である。
したがって環境が報告する trajectory-best score と、全出力を公式 scorer で採点した最終
状態のscoreは一致しない場合がある。drawdown plane の追加により観測は44 planesとなり、
43 planesで学習した既存checkpointとは互換性がない。

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
