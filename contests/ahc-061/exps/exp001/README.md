# exp001: AHC061 PPO のwidth × learning rate sweep

## 結論

21 run（channels 3種類 × LR 7種類）の比較では、最良LRはwidthの増加とともに低LR側へ移動した。

| channels | 最良LR | final mean_score |
| ---: | ---: | ---: |
| 64 | 7.114851130e-4 | 159,214.51 |
| 128 | 5.334838230e-4 | 163,300.91 |
| 256 | 4.000563416e-4 | 164,060.19 |

通常のparameterizationでは、widthを変えてもグローバルLRを共通にすると更新量がwidth依存になりやすい。この結果は「width増加に伴って最適LRが小さくなる」というStandard Parameterizationで自然な傾向と整合する。一方で、各設定は1 seedのみであり、scoreのばらつき・探索範囲の粗さがあるため、厳密な指数則やscaling lawを主張するには不十分である。

## 目的

PPOのConvNeXt幅 `channels ∈ {64, 128, 256}` に対して、学習率の転移性を確認する。基準LRを `3e-4` とし、対数等間隔の初期5点と、その高LR側の隣接点の対数中点2点を評価した。

## 実験条件

- 実行: task 287–307、すべてSuccess
- 学習: 10,000,000 environment stepsを指定。rollout単位への丸めにより最終ログは10,035,200 steps
- seed: `seed_start=0`, `seed_stride=1`
- 並列環境数: 1,024
- PPO epochs: 1、minibatch size: 1,024
- モデル: ConvNeXt、trunk blocks=2。value headは追加でConvNeXt blockを2個持つ
- channels: 64 / 128 / 256
- LR: `9.486832981e-5`, `1.687023976e-4`, `3.000000000e-4`, `4.000563416e-4`, `5.334838230e-4`, `7.114851130e-4`, `9.486832981e-4`
- それ以外の設定は [config/ppo_train.toml](config/ppo_train.toml) を参照

`final mean_score` は各runの最終ログに記録された W&B の `episode/score_mean` を用いた。

## 結果

| task | channels | LR | final mean_score | final env steps | elapsed sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 287 | 64 | 9.486832981e-5 | 115230.48 | 10,035,200 | 1762.3 |
| 288 | 64 | 1.687023976e-4 | 131348.13 | 10,035,200 | 1692.7 |
| 289 | 64 | 3.000000000e-4 | 144705.20 | 10,035,200 | 1698.5 |
| 302 | 64 | 4.000563416e-4 | 150573.26 | 10,035,200 | 1741.2 |
| 290 | 64 | 5.334838230e-4 | 158069.21 | 10,035,200 | 1692.2 |
| 303 | 64 | 7.114851130e-4 | 159214.51 | 10,035,200 | 1737.1 |
| 291 | 64 | 9.486832981e-4 | 155164.31 | 10,035,200 | 1691.5 |
| 292 | 128 | 9.486832981e-5 | 126919.88 | 10,035,200 | 1767.4 |
| 293 | 128 | 1.687023976e-4 | 144280.03 | 10,035,200 | 1760.4 |
| 294 | 128 | 3.000000000e-4 | 154733.23 | 10,035,200 | 1760.4 |
| 304 | 128 | 4.000563416e-4 | 158538.88 | 10,035,200 | 1831.2 |
| 295 | 128 | 5.334838230e-4 | 163300.91 | 10,035,200 | 1758.9 |
| 305 | 128 | 7.114851130e-4 | 159398.91 | 10,035,200 | 1885.1 |
| 296 | 128 | 9.486832981e-4 | 157570.85 | 10,035,200 | 1756.1 |
| 297 | 256 | 9.486832981e-5 | 143310.78 | 10,035,200 | 2024.2 |
| 298 | 256 | 1.687023976e-4 | 152634.12 | 10,035,200 | 2023.5 |
| 299 | 256 | 3.000000000e-4 | 163321.35 | 10,035,200 | 2026.2 |
| 306 | 256 | 4.000563416e-4 | 164060.19 | 10,035,200 | 2182.0 |
| 300 | 256 | 5.334838230e-4 | 164003.62 | 10,035,200 | 2024.0 |
| 307 | 256 | 7.114851130e-4 | 161997.54 | 10,035,200 | 2193.0 |
| 301 | 256 | 9.486832981e-4 | 162128.42 | 10,035,200 | 2026.9 |

- LR–最終score: [figures/lr_vs_final_mean_score.html](figures/lr_vs_final_mean_score.html)
- FLOPs–学習曲線: [figures/flops_vs_mean_score.html](figures/flops_vs_mean_score.html)
- 生の全履歴: [data/runs.json](data/runs.json)
- 最終値のみのCSV: [data/final_scores.csv](data/final_scores.csv)

## FLOPs見積もり

PyTorchの `torch.utils.flop_counter.FlopCounterMode` で、盤面10×10・batch size 1の `ActorCritic` を実行して計測した。各environment stepに対して、rollout時のforwardを1回、PPO更新時のforward+backwardを1回として合算した。optimizer更新・環境シミュレーション・GAE・データ転送は含めない。

| channels | rollout forward / step | update forward+backward / step | 合計 / step | 10.0352M steps時 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 45.4M | 147.5M | 193.0M | 1.94e15 |
| 128 | 145.1M | 516.8M | 661.9M | 6.64e15 |
| 256 | 507.2M | 1.921G | 2.428G | 2.44e16 |

FLOPs図の横軸は、上記のstepあたり合計値に `summary/cumulative_env_steps` を掛けた累積値である。

## 考察

1. **LR transferは成立していない。** channels 64では約7.11e-4、128では約5.33e-4、256では約4.00e-4が最良で、最適LRはwidthに応じて下がっている。
2. **大きいwidthは低LRで有利だが、改善幅は逓減している。** 最高scoreは64→128で約4.1k、128→256で約0.8k上昇した。計算量はそれぞれ約3.4倍・3.7倍である。
3. **μPを導入する動機がある。** μPは層・パラメータ種別ごとに初期化とLRスケールを設計し、width間で最適hyperparameterを転移させることを狙う。単一のグローバルLRをwidth依存で手調整するより、width sweepを効率化できる可能性がある。
4. **次の検証。** 最良LR近傍を複数seedで再実行し、mean±標準誤差を比較する。その後、ConvNeXt trunk・policy head・value headを含めたμP対応のparameter groupを設計して、width間で同一LRが使えるか再検証する。

## 再現データ

`data/runs.json` はtask ID、W&B run ID、実行時刻、最終値、および各PPO updateの `env_steps` と `mean_score` を保存している。W&B summaryとローカルの `.wandb` historyから抽出したため、図表はこのファイルだけで再生成できる。
