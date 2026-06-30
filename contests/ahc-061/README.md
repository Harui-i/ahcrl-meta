# AHC-template

優勝しよう。
コンテスト url: <https://atcoder.jp/contests/ahc061>

問題文 url: <https://atcoder.jp/contests/ahc061/tasks/ahc061_a>
問題文: [problem_en.md] / [problem_ja.md]

## コード配置

- `tools/`: AtCoder公式toolsのvendor snapshot。原則として編集しない。
- `rl-tools/`: PPO訓練用の自前Rust補助crate。公式tools crateに依存しつつ、公式側のprivateロジックが必要な箇所は `official_compat` に分離してparity testで検証する。

## PPO訓練

```bash
uv run python3 -m ahcrl.contests.ahc061.train_ppo --config contests/ahc-061/configs/ppo_smoke.toml
```

設定はTOMLの `[train]` に書く。CLIで同じオプションを指定した場合はCLI側を優先する。
学習開始時には解決済みconfigを `config={...}` のJSON形式でstdoutへ出す。
モデル規模は `model_channels` と `model_blocks` で指定する。
