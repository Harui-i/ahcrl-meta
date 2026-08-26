# ahcrl-meta

## 開発コマンド

Python の開発用コマンドは `make` 経由で実行できます（依存関係は `uv` が管理します）。

```bash
make ruff          # lint
make pyright       # 型検査
make test          # テスト
make check         # 上記とフォーマット検証を一括実行
make format        # コードを整形
```

## 新しいAHCの初期化

共通のディレクトリ、PPO設定、Pahcer設定、TorchScript/C++評価の雛形を作るには、例えば次を実行します。

```bash
make contest-init ahc068
```

公式toolsは自動取得せず、`contests/ahc-068/tools/` に配置します。URLを明示的に指定したい場合だけ、次のように実行できます。

```bash
make contest-init ahc068 TOOLS_URL="https://example.invalid/tools.zip"
```

既存ファイルは上書きされません。問題文、公式tools、環境・行動定義、モデル、PPOエントリポイントはコンテストごとに実装します。
