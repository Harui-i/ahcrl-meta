# AHC068

新しいAHCの作業ディレクトリ。

- コンテスト: https://atcoder.jp/contests/ahc068
- 問題文: [problem_ja.md](problem_ja.md) / [problem_en.md](problem_en.md)
- 公式tools: `tools/`（取得後は原則編集しない）
- PPO設定: `configs/`
- Pahcer設定: `eval/pahcer_config.toml`
- PPO成果物: `artifacts/ppo/`（git管理外）

## 最初にすること

1. `problem_ja.md` と `problem_en.md` に問題文を保存する。
2. `tools/` にAtCoder公式toolsを配置し、`cargo build --release`する。
3. `src/ahcrl/contests/ahc068/` のencoder・simulator・modelを実装する。
4. `eval/main.cpp`を公式visで検証する。
5. `make check`を通してからPPOを開始する。
