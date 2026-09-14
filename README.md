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

## Torch Modula実験

PPOの既定optimizerは従来どおりAdamWです。AHC061/063で、モデルに宣言された自然な
重み幾何を使うhybrid optimizerを試すには、設定の`[training]`で次を指定します。

```toml
optimizer = "modula"
weight_decay = 0.01
modula_momentum = 0.95
modula_nesterov = true
modula_diagnostics_interval = 100
modula_initialize = true
modula_project = true
```

`modula_initialize`と`modula_project`は独立に切り替えられます。Modula weightには
Muon型のmomentumとJiachengの固定six-step Newton--Schulz dualizationを適用します。
反復回数は設定項目ではありません。bias、normalizationのaffine parameter、LayerScaleは
AdamWで更新します。通常Convはgroupごとにkernelを
行列化し、depthwise Convはchannelごとのfilterとして扱います。このConv幾何は空間を
含む厳密な畳み込み作用素ノルムではなく、最適化実験用のkernel-matrix近似です。
高コストな直交性・spectral norm診断は`modula_diagnostics_interval` optimizer stepごとに
計算します。

新しいtrainable layerを追加する場合は`ModularLinear`などの意味付きModuleを使うか、
AdamW対象として明示的に登録する必要があります。未分類のtrainable parameterがある
モデルではModula optimizerの構築が失敗します。
