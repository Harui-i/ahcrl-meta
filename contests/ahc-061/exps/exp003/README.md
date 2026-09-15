# exp003: AHC061 Modula のwidth × learning rate sweep

## 結論

Modula optimizer では、channels 64 の最良LRは `0.5333`、channels 128 / 256 の最良LRはともに `0.4` だった。探索した4点のうち `0.4` は64以外で最良、64でも最良値との差は1.81%であり、exp001の通常PPOよりwidthをまたぐLR転移性は良好に見える。

| channels | 最良LR | final mean_score |
| ---: | ---: | ---: |
| 64 | 0.5333 | 175,897.57 |
| 128 | 0.4 | 178,979.12 |
| 256 | 0.4 | **181,501.44** |

単一seed・LR 4点の探索であり、厳密な最適LRや小さなscore差の優劣は主張しない。最良設定は channels=256 / LR=0.4 で、exp001の通常PPO最良値（channels=256 / LR=`4.000563416e-4`、164,060.19）より17,441.25（+10.63%）高かった。ただしoptimizer以外にも学習step数が10Mから20Mへ異なるため、この差をModula単独の効果とは解釈しない。

## 目的

Modula optimizerで、ConvNeXt幅 `channels ∈ {64, 128, 256}` に対するlearning rateの転移性を調べる。LRは `0.3`, `0.4`, `0.5333`, `0.7111` とした。

## 実験条件

- 実行: Pueue task 326–337、全条件Success
- 学習: `total_steps=20,000,000` を指定。rollout境界への丸めにより最終値は20,070,400 environment steps
- seed: `seed_start=0`, `seed_stride=1`
- 並列環境数: 1,024、rollout steps: 100
- PPO: epochs=1、minibatch size=1,024、clip=0.2
- optimizer: Modula（weight decay=0.01、momentum=0.95、Nesterov、target KL=0.03）
- model: ConvNeXt、channels=64 / 128 / 256、blocks=2
- W&B tags: `ahc061`, `ppo`, `modula`, `jiacheng-ns6`, `exp-003`
- ベース設定: [ppo_modula.toml](../../configs/ppo_modula.toml)

`final mean_score` は各runの最終更新でW&Bに記録された `episode/score_mean` である。

## 結果

| task | channels | LR | final mean_score | final env steps | elapsed sec | W&B run |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 326 | 64 | 0.3 | 170,737.90 | 20,070,400 | 1554.8 | [4luklyvs](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/4luklyvs) |
| 327 | 64 | 0.4 | 172,714.10 | 20,070,400 | 1550.5 | [15a5w5fb](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/15a5w5fb) |
| 328 | 64 | 0.5333 | **175,897.57** | 20,070,400 | 1454.8 | [nzl45zty](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/nzl45zty) |
| 329 | 64 | 0.7111 | 171,694.33 | 20,070,400 | 1413.6 | [1xbn8z2j](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/1xbn8z2j) |
| 330 | 128 | 0.3 | 177,025.60 | 20,070,400 | 1600.6 | [n5qxjl7v](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/n5qxjl7v) |
| 331 | 128 | 0.4 | **178,979.12** | 20,070,400 | 1586.8 | [aeg6e8op](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/aeg6e8op) |
| 332 | 128 | 0.5333 | 176,833.62 | 20,070,400 | 1600.2 | [3fxwkz5u](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/3fxwkz5u) |
| 333 | 128 | 0.7111 | 175,554.60 | 20,070,400 | 1338.3 | [jcbnykzr](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/jcbnykzr) |
| 334 | 256 | 0.3 | 179,474.48 | 20,070,400 | 2148.8 | [cwh743aa](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/cwh743aa) |
| 335 | 256 | 0.4 | **181,501.44** | 20,070,400 | 2134.8 | [zp6rvg1a](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/zp6rvg1a) |
| 336 | 256 | 0.5333 | 176,656.23 | 20,070,400 | 2128.1 | [uqd7d5an](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/uqd7d5an) |
| 337 | 256 | 0.7111 | 176,350.21 | 20,070,400 | 1792.1 | [6wxvl78n](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/6wxvl78n) |

- LR–最終score: [figures/lr_vs_final_mean_score.html](figures/lr_vs_final_mean_score.html)
- FLOPs–学習曲線: [figures/flops_vs_mean_score.html](figures/flops_vs_mean_score.html)
- 生の全履歴: [data/runs.json](data/runs.json)
- 最終値のみのCSV: [data/final_scores.csv](data/final_scores.csv)

## FLOPs見積もり

exp001と同じ、PyTorch `FlopCounterMode` によるモデルforward/backwardの見積もりを使う。各environment stepにつきrollout forwardを1回、PPO更新のforward+backwardを1回とし、optimizer更新・環境シミュレーション・GAE・データ転送は含めない。Modula固有のoptimizer演算もこの見積もりからは除外している。

| channels | FLOPs / step | 20.0704M steps時 |
| ---: | ---: | ---: |
| 64 | 193.0M | 3.87e15 |
| 128 | 661.9M | 1.33e16 |
| 256 | 2.428G | 4.87e16 |

## 考察

1. **`LR=0.4` はwidth間で実用的に転移した。** channels 128 / 256 で最高、64でも最良の`0.5333`との差は3,183.48に留まった。exp001の通常PPOでは最良LRが64→128→256で約`7.11e-4`→`5.33e-4`→`4.00e-4`と下がったのに比べ、今回の依存性は弱い。
2. **width増加は依然として有利だが、LR感度は大きい。** 各widthで0.4付近が最良であり、256では0.4から0.5333への上昇で4,845.22低下した。もっと高いLRを広く試す前に、0.35–0.50の狭い範囲を複数seedで比較する価値がある。
3. **次の検証。** channels=128 / 256、LR=0.4を中心に複数seedでmean±標準誤差を出す。その後、同一学習step・seedで標準AdamW PPOとの対照を置き、optimizer変更の寄与を切り分ける。

## 再現データ

`data/runs.json` にはtask ID、W&B run ID、解決済み条件、最終値、各PPO updateの環境stepとmean scoreを保存している。`collect_runs.py` を実行すると、W&B APIからこのJSON・CSV・図表を再生成できる。
