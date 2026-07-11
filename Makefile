.DEFAULT_GOAL := help

.PHONY: help ruff ruff-fix format format-check pyright test check

help: ## 利用可能なコマンドを表示する
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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
