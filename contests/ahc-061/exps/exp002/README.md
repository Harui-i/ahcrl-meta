# exp002: Advantage-Adaptive Clipping for AHC061 PPO

## 結論

κ=0.15 / 0.20 とも固定 clipping の score を下回った。adaptive 条件内では κ=0.20 が最良だったが、固定 PPO 比で 2.84% 低い。

| condition | κ | final mean score | fixed PPO 比 | W&B run |
| --- | ---: | ---: | ---: | --- |
| fixed PPO (exp001 task 306) | — | 164,060.19 | — | [ok5n1134](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/ok5n1134) |
| adaptive clipping | 0.15 | 158,565.69 | -5,494.50 (-3.35%) | [z8i96ybh](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/z8i96ybh) |
| adaptive clipping | 0.20 | 159,401.37 | -4,658.82 (-2.84%) | [255qpitp](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/255qpitp) |

- 最終 score: [figures/final_mean_score.html](figures/final_mean_score.html)
- 最終 adaptive 診断値: [figures/adaptive_clipping_diagnostics.html](figures/adaptive_clipping_diagnostics.html)
- 集計値: [data/final_scores.csv](data/final_scores.csv) / [data/final_diagnostics.csv](data/final_diagnostics.csv)

単一 seed の探索的比較であり、統計的な優劣は主張しない。ただし、この条件では adaptive clipping を採用する根拠は得られなかった。次に試すなら κ=0.20 近傍の再現性を複数 seed で確認するより、positive-side only と PPO epochs 増加を分離して比較する。

## 仮説

rollout batch で標準化した advantage を \(z_i\) とし、sample ごとの PPO clipping 幅を

\[
\epsilon_i = \operatorname{clamp}(\kappa |z_i|, 0.05, 0.40)
\]

とする。大きい相対 advantage には広い更新幅を、ほぼゼロの advantage には下限の更新幅を与える。状態条件付き advantage 分散は観測できないため、batch 標準化はその近似である。

## 固定条件

- 10,000,000 environment steps（rollout 境界で 10,035,200 steps まで進む）
- seed: `seed_start=0`, `seed_stride=1`
- `num_envs=1024`、`rollout_steps=100`、PPO epochs=1、minibatch size=1024
- model: channels=256、blocks=2
- learning rate: `4.000563416e-4`
- W&B tags: `ahc061`, `ppo`, `exp-002`, `adaptive-clip`
- task 308（κ=0.15）: 2026-09-14 15:42:39–15:57:53 JST
- task 309（κ=0.20）: 2026-09-14 15:57:53–16:13:07 JST

## 実装履歴

実験で使用した adaptive clipping の実装・テスト・実行設定は commit [`9a7ad2f`](../../../../commit/9a7ad2f) に保存している。現行 trainer からは削除済みである。
