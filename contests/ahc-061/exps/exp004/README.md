# exp004: AHC061 Modula + Proximal EWMA の learning rate sweep

## 結論

この単一seed・100M-step探索では、**LR=0.303143313** が最良で、final mean score は **197,747.83** だった。次点のLR=0.348220225より1,834.20（0.94%）、基準のLR=0.4より4,037.79（2.08%）高い。探索範囲では0.30近傍に山があり、0.4まで上げると低下した。

単一seedの小差であるため、この結果だけで厳密な最適LRを断定はしない。次はLR=0.28–0.34を複数seedで再検証するのが妥当である。

## 目的

`ppo_modula_ewma.toml` をベースに、channels=64・100M environment steps の条件でlearning rateの適切な範囲を探索する。指定範囲 `[0.2, 0.4]` を両端を含む対数等間隔の6点に分割した。

## 固定条件

- ベース設定: [ppo_modula_ewma.toml](../../configs/ppo_modula_ewma.toml)
- 上書き: `model_channels=64`, `env_workers=12`, `total_steps=100,000,000`
- optimizer: Modula + Proximal EWMA (`proximal_ewma_com=256`)
- `num_envs=1024`, `rollout_steps=100`, PPO epochs=1, minibatch size=1024
- seed: `seed_start=0`, `seed_stride=1`
- W&B tags はベース設定を継承する。run name は `ppo-modula-lr{LR}-ch64-ewma256-sep1-100M-2` とする。

## 結果

LRは \(0.2 \times 2^{i/5}\)（\(i=0,\ldots,5\)）である。全runは100,044,800 environment stepsまで完走した。

| task | LR | final mean score | elapsed sec | W&B run |
| ---: | ---: | ---: | ---: | --- |
| 343 | 0.2 | 195,281.79 | 6448.5 | [vp8klbbo](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/vp8klbbo) |
| 344 | 0.229739671 | 194,654.04 | 6257.5 | [kxayn2ia](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/kxayn2ia) |
| 345 | 0.263901582 | 195,178.91 | 6082.5 | [fv81dx1t](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/fv81dx1t) |
| 346 | 0.303143313 | **197,747.83** | 5919.5 | [ablick03](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/ablick03) |
| 347 | 0.348220225 | 195,913.64 | 5811.5 | [7arwho3p](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/7arwho3p) |
| 348 | 0.4 | 193,710.04 | 5444.8 | [a2j921uf](https://wandb.ai/harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061/runs/a2j921uf) |

- LR–最終score: [figures/lr_vs_final_mean_score.html](figures/lr_vs_final_mean_score.html)
- 学習曲線: [figures/steps_vs_mean_score.html](figures/steps_vs_mean_score.html)
- 生の全履歴: [data/runs.json](data/runs.json)
- 最終値のみのCSV: [data/final_scores.csv](data/final_scores.csv)

## 考察

1. **0.30近傍が最良だった。** 0.2から0.303143313まで改善し、さらに増やすと悪化した。0.263901582から0.303143313への差は2,568.92（1.32%）で、狭い再探索を行う価値がある。
2. **学習速度と最終scoreにはトレードオフがある。** LR=0.4は最短の5,444.8秒で完走したが最終scoreは最良値より2.08%低い。LR=0.303143313は5919.5秒で、実行時間を8.7%増やしてscoreを改善した。
3. **exp003との直接比較はしない。** exp003はEWMAなし・20M stepsであり、本実験はEWMAあり・100M stepsのため、最終scoreの差をLRまたはEWMA単独の効果として解釈できない。

## 実行コマンド

```bash
uv run python3 -m ahcrl.contests.ahc061.train_ppo \
  --config contests/ahc-061/configs/ppo_modula_ewma.toml \
  --model-channels 64 --lr <LR> --env-workers 12 \
  --wandb-name <W&B name> --total-steps 100000000
```

`collect_runs.py` を実行すると、W&B APIからこのJSON・CSV・図表を再生成できる。
