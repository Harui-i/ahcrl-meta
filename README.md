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
