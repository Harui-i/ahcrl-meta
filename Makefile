.DEFAULT_GOAL := help

.PHONY: help ruff ruff-fix format format-check pyright test check \
	contest-init

help: ## 利用可能なコマンドを表示する
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

contest-init: ## 新しいAHCの共通ディレクトリと設定雛形を作る
	@python3 scripts/contest_init.py "$(if $(CONTEST),$(CONTEST),$(word 2,$(MAKECMDGOALS)))" $(if $(TOOLS_URL),--tools-url "$(TOOLS_URL)",)

# `make contest-init ahc068` の ahc068 を make の独立したゴールとして
# 扱えるようにするための no-op パターンゴール。実際の処理は上のレシピで行う。
ahc%:
	@:

ruff: ## Ruff による lint を実行する
	uv run ruff check .

ruff-fix: ## Ruff の自動修正を適用する
	uv run ruff check . --fix

format: ## Ruff でコードを整形する
	uv run ruff format .

format-check: ## コードフォーマットを検証する
	uv run ruff format . --check

pyright: ## Pyright による型検査を実行する
	uv run pyright

test: ## Pytest を実行する
	uv run pytest

check: ruff format-check pyright test ## lint・format・型検査・テストを一括実行する
