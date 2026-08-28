.DEFAULT_GOAL := help

.PHONY: help ruff ruff-fix format format-check pyright test rust-format-check \
	rust-clippy rust-test rust-check contest-check check contest-init

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

rust-format-check: ## 共通EnvとAHC063 Rustコードのフォーマットを検証する
	cargo fmt --check --manifest-path crates/ahcrl-env-core/Cargo.toml
	rustfmt --check contests/ahc-063/tools/src/rl_bridge.rs
	cargo fmt --check --manifest-path contests/ahc-063/rl-tools/Cargo.toml

rust-clippy: ## 共通EnvとAHC063 RustコードをClippyで検証する
	cargo clippy --manifest-path crates/ahcrl-env-core/Cargo.toml --all-targets --no-deps -- -D warnings
	cargo clippy --manifest-path contests/ahc-063/rl-tools/Cargo.toml --all-targets --no-deps -- -D warnings

rust-test: ## 共通EnvとAHC063 Rustコードをテストする
	cargo test --manifest-path crates/ahcrl-env-core/Cargo.toml
	cargo test --manifest-path contests/ahc-063/rl-tools/Cargo.toml

rust-check: rust-format-check rust-clippy rust-test ## Rustのformat・lint・testを一括実行する

contest-check: ## 指定AHCのrl-toolsを検証する。例: make contest-check CONTEST=ahc-068
	@test -n "$(CONTEST)" || (echo "CONTEST is required, e.g. CONTEST=ahc-068" >&2; exit 2)
	@test -f "contests/$(CONTEST)/rl-tools/Cargo.toml" || (echo "rl-tools manifest not found for $(CONTEST)" >&2; exit 2)
	cargo fmt --check --manifest-path contests/$(CONTEST)/rl-tools/Cargo.toml
	cargo clippy --manifest-path contests/$(CONTEST)/rl-tools/Cargo.toml --all-targets --no-deps -- -D warnings
	cargo test --manifest-path contests/$(CONTEST)/rl-tools/Cargo.toml

check: ruff format-check pyright test rust-check ## lint・format・型検査・テストを一括実行する
