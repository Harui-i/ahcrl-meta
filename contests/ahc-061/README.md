# AHC-template

優勝しよう。
コンテスト url: <https://atcoder.jp/contests/ahc061>

問題文 url: <https://atcoder.jp/contests/ahc061/tasks/ahc061_a>
問題文: [problem_en.md] / [problem_ja.md]

## コード配置

- `tools/`: AtCoder公式toolsのvendor snapshot。原則として編集しない。
- `rl-tools/`: PPO訓練用の自前Rust補助crate。公式tools crateに依存しつつ、公式側のprivateロジックが必要な箇所は `official_compat` に分離してparity testで検証する。

## PPO訓練

AHC061 の標準訓練方式は PPO-EWMA であり、標準設定は
`proximal_ewma = true` と `proximal_ewma_com = 256.0` を指定する。通常 PPO との
比較時だけ `--no-proximal-ewma` を使用する。

```bash
uv run python3 -m ahcrl.contests.ahc061.train_ppo --config contests/ahc-061/configs/ppo_train.toml
```

設定はTOMLの名前空間付きセクション（`[training]`、`[ppo]` など）に書く。CLIで同じ
オプションを指定した場合はCLI側を優先する。
学習開始時には解決済みconfigを `config={...}` のJSON形式でstdoutへ出す。
モデル規模は `model_channels` と `model_blocks` で指定する。
checkpoint保存頻度は `checkpoint_interval_updates` で「何updateごとに保存するか」を指定する。最後のupdateは必ず保存する。
新規runは `artifact_dir` 配下に `run_*` ディレクトリを作って保存する。
同一runを継続する場合は `--resume-dir contests/ahc-061/artifacts/ppo/run_...` を指定する。resume時は保存済みconfigを使い、`--total-steps` だけ増やせる。したがって既存の通常 PPO run も設定を変えずに再開できる。
別runの初期値としてモデルだけ読む場合は `--init-checkpoint path/to/checkpoint.pt` を指定する。
