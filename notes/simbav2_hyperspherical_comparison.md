# SimbaV2 の hyperspherical 系実装との比較メモ

対象:
- ローカル実装: [`src/ahcrl/nn/blocks.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:9)
- ローカル利用箇所: [`src/ahcrl/contests/ahc061/model.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/model.py:8), [`src/ahcrl/contests/ahc061/train_ppo.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:67)
- 参照実装: <https://github.com/DAVIAN-Robotics/SimbaV2>
- 論文: <https://arxiv.org/abs/2502.15280>

このメモは、`model_block_type = "spherical_convnext"` を選んだ場合の trunk が SimbaV2 の推奨構造にどの程度寄っているかを整理する。
policy/value head が異なること、PPO の scalar value であり SimbaV2 公式の distributional critic ではないことは、このメモでは許容差分として扱う。

## 結論

`spherical_convnext` 前提では、trunk の hyperspherical 構造は **SimbaV2 にかなり近い**。

特に次は公式実装とよく対応している。

1. `HyperEmbedder2d` は `concat const -> l2 norm -> bias-free Linear -> Scaler -> l2 norm` になっている。
2. `SphericalConvNeXtBlock` は `HyperMLP -> residual + alpha * (transform - residual) -> l2 norm` になっており、公式の `HyperLERPBlock` に近い。
3. `spherical_convnext` の学習時は optimizer step 後に hyperspherical weight projection が走る。

一方、SimbaV2 の安定化パッケージとして見ると、主な未再現点は次。

1. 観測の running mean/std normalization、つまり RSNorm / `ObservationNormalizer` 相当がない。
2. reward scaling が公式実装と違う。公式は discounted return の running variance と `g_max` 下限を使うが、ローカルは即時報酬 std の簡易版で、しかもデフォルト無効。
3. 現在の config は `model_block_type = "convnext"` なので、デフォルト実験ではこの SimbaV2 風 trunk 自体を使っていない。

つまり、`spherical_convnext` の trunk だけを見れば「SimbaV2 風」と呼んでよい水準まで来ている。次に寄せるべきズレは block 内部ではなく、**観測 RSNorm と reward scaling**。

## SimbaV2 側の構造

参照実装の hyperspherical 系は主に次にまとまっている。

- `scale_rl/agents/simbaV2/simbaV2_layer.py`
- `scale_rl/agents/simbaV2/simbaV2_network.py`
- `scale_rl/agents/simbaV2/simbaV2_update.py`
- `scale_rl/agents/wrappers/normalization.py`

公式実装の流れは大まかに次。

```text
ObservationNormalizer
-> HyperEmbedder
   -> concat c_shift
   -> l2normalize
   -> HyperDense
   -> Scaler
   -> l2normalize
-> HyperLERPBlock x N
   -> HyperMLP
      -> HyperDense
      -> Scaler
      -> ReLU + eps
      -> HyperDense
      -> l2normalize
   -> residual + alpha_scaler * (mlp(x) - residual)
   -> l2normalize
-> predictor
```

学習更新後は `l2normalize_network` により `hyper_dense` の kernel が再投影される。

reward 側は、`RewardNormalizer` が discounted return `G` の running variance を追跡し、報酬をそのスケールで割る。さらに `normalized_g_max` に基づく下限がある。

参照:
- [`HyperEmbedder`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L68-L86)
- [`HyperMLP`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L46-L65)
- [`HyperLERPBlock`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L89-L117)
- [`l2normalize_network`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_update.py#L24-L46)
- [`RewardNormalizer`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/wrappers/normalization.py)

## ローカル実装との対応

### `Scaler`

ローカルの [`Scaler`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:188) は、公式実装とほぼ同じ。

公式:

```text
scaler parameter initial value = scale
forward multiplier = init / scale
output = scaler * forward_scaler * x
```

ローカル:

```text
scaler = nn.Parameter(full((dim,), scale))
forward_scaler = init / scale
output = x * scaler * forward_scaler
```

形状チェックや `scale != 0` の検証が追加されている程度で、設計上のズレは小さい。

### `l2_normalize`

ローカルの [`l2_normalize`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:11) は、公式の `l2normalize` と同じ役割。

公式は `norm = max(l2norm, EPS)`、ローカルは `sqrt(clamp(sum(x^2), eps^2))` なので、下限の置き方は実装詳細レベルの差分。

### `HyperEmbedder2d`

ローカルの [`HyperEmbedder2d`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:324) は、公式の `HyperEmbedder` とかなり対応している。

ローカル:

```text
NCHW planes
-> NHWC
-> ShiftL2Norm(c=3.0)
   -> concat const
   -> l2_normalize
-> HyperLinear(in_channels + 1, out_channels)
-> Scaler(init=sqrt(2 / out_channels), scale=sqrt(2 / out_channels))
-> l2_normalize
-> NCHW
```

公式:

```text
x
-> concat c_shift
-> l2normalize
-> HyperDense
-> Scaler
-> l2normalize
```

`ShiftL2Norm` 単体は公式 embedder の前半だけだが、`HyperEmbedder2d` として使う場合は後段の `Linear + Scaler + l2 norm` まで含むので、embedder 全体としてはかなり近い。

### `HyperMLP`

ローカルの [`HyperMLP`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:284) は、公式の `HyperMLP` とよく対応している。

共通点:
- bias-free な hyperspherical linear を2層使う
- 1層目の後に `Scaler`
- `ReLU + eps` でゼロベクトルを避ける
- 2層目の後に `l2_normalize`
- expansion 分だけ hidden を広げる

scaler の default も、`sqrt(2 / channels) / sqrt(expansion)` で、公式 `HyperLERPBlock` が `scaler_init / sqrt(expansion)` を渡す構造と対応している。

### `SphericalConvNeXtBlock`

ローカルの [`SphericalConvNeXtBlock`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:367) は、名前は `ConvNeXt` のままだが、現行実装は公式の `HyperLERPBlock` に近い。

ローカル:

```text
residual = x
y = HyperMLP(x)
y = residual + alpha_scaler(y - residual)
y = l2_normalize(y)
```

公式:

```text
residual = x
x = HyperMLP(x)
x = residual + alpha_scaler(x - residual)
x = l2normalize(x)
```

`alpha_init` は [model.py](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/model.py:119) で `1 / (blocks + 1)`、`alpha_scale` は block 側の default で `1 / sqrt(channels)`。これは公式 config の `alpha_init` / `alpha_scale` と対応している。

古いメモでは「`SphericalConvNeXtBlock` は HyperLERPBlock ではない」と書いていたが、現行コードではその指摘はもう古い。

### 重み投影

公式 SimbaV2 は更新後に `hyper_dense` の重みを l2 normalize する。

ローカルでは [`project_hyperspherical_weights_`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:270) があり、[train_ppo.py](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:598) で optimizer step 後に呼ばれる。

```python
if (
    args.weight_projection
    or getattr(args, "model_block_type", "") == "spherical_convnext"
):
    project_hyperspherical_weights_(grad_model)
```

したがって、`spherical_convnext` の場合は `--weight-projection` を明示しなくても projection が走る。ここも公式に寄っている。

注意点として、ローカルの projection 対象は `HyperLinear` のみ。公式の `regex="hyper_dense"` と意図は近いが、命名ではなく型で対象を選んでいる。

## 観測前処理のズレ

SimbaV2 公式 config では `normalize_observation: true` で、agent wrapper の `ObservationNormalizer` が入る。

公式の `ObservationNormalizer` は、観測座標ごとの running mean/std を持ち、

```text
(observation - running_mean) / sqrt(running_var + eps)
```

を actor/critic に渡す。

一方、ローカル AHC061 では、Rust encoder が plane を手作業でスケールしている。

例:
- `M / MAX_PLAYERS`
- `U / MAX_LEVEL`
- score ratio
- distance や座標を board size で割る
- legal mask や one-hot 系 plane

これは入力値のレンジを整える意味では有用だが、公式 SimbaV2 の RSNorm / `ObservationNormalizer` とは別物。

したがって `spherical_convnext` の実際の流れは次。

```text
manual-scaled planes
-> concat const
-> l2 norm
-> HyperLinear
-> Scaler
-> l2 norm
-> HyperLERP-like blocks
```

公式推奨により近い流れは次。

```text
manual-scaled or raw-ish planes
-> running mean/std observation normalization
-> concat const
-> l2 norm
-> HyperLinear
-> Scaler
-> l2 norm
-> HyperLERP blocks
```

つまり、ユーザーの記憶していた `RSNorm + concat const + L2Norm + Linear + Scaler + ...` で言うと、**`concat const` 以降は概ね入っているが、RSNorm がない**。

AHC の観測は plane ごとに意味が強く違うため、RSNorm を入れるなら少なくとも plane/channel ごとの running mean/std にするのが自然。空間位置ごとの統計まで持つか、channel 単位に集約するかは実験対象。

## Reward Scaling のズレ

公式 SimbaV2 config では `normalize_reward: true`、`normalized_g_max: 5.0`。

公式の `RewardNormalizer` は、即時報酬そのものではなく discounted return の running estimate `G` を更新する。

```text
G = gamma * (1 - done) * G + reward
G_rms.update(G)
G_r_max = max(G_r_max, max(abs(G)))
denominator = max(sqrt(G_rms.var + eps), G_r_max / g_max)
scaled_reward = reward / denominator
```

ローカルの [`RunningRewardScaler`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:67) は、rollout 内の即時報酬を running std で割る簡易版。

```text
running std over immediate rewards
scaled_reward = reward / max(std, eps)
```

さらに default は [train_ppo.py](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:44) で `reward_scale_mode = "none"` なので無効。

したがって、`reward_scale_mode = "running_std"` を有効にしても、公式 SimbaV2 の reward scaling とは一致しない。

PPO では advantage normalization や value loss の扱いも絡むため、SAC 用の公式設計をそのまま移植すればよいとは限らない。ただし「SimbaV2 の reward scaling に寄せる」という目的なら、次が必要。

1. discounted return `G` の running variance を使う
2. `normalized_g_max` に相当する下限を入れる
3. scaler state を checkpoint に保存する
4. scaled reward と unscaled reward の logging を分ける

## デフォルト config の注意

現在の [ppo_train.toml](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/contests/ahc-061/configs/ppo_train.toml:23) は:

```toml
model_block_type = "convnext"
```

なので、通常の設定では `spherical_convnext` は使われない。

`spherical_convnext` を評価するなら、少なくとも config または CLI で次を明示する必要がある。

```toml
model_block_type = "spherical_convnext"
```

この場合、weight projection は自動で有効になる。

## 実装上の含意

`spherical_convnext` 前提で、SimbaV2 trunk の再現度を上げる優先順位は次。

1. 観測 RSNorm / running mean-std normalization を追加する
2. reward scaling を公式の discounted-return RMS + `g_max` 下限に寄せる
3. `spherical_convnext` という名前を `simbav2_trunk` などに整理するか検討する
4. config に SimbaV2 風実験用 preset を作る

逆に、次はすでに大きな問題ではない。

- `HyperEmbedder2d` の `concat const -> L2 -> Linear -> Scaler -> L2`
- `SphericalConvNeXtBlock` の LERP 構造
- `HyperMLP` の `ReLU + eps` と最終 L2Norm
- `spherical_convnext` 使用時の weight projection

残差は「trunk の部品」よりも「trunk に入る前後の正規化と学習系」にある。

## 参照したローカル箇所

- [`src/ahcrl/nn/blocks.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:9)
- [`src/ahcrl/contests/ahc061/model.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/model.py:8)
- [`src/ahcrl/contests/ahc061/train_ppo.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:67)
- [`contests/ahc-061/configs/ppo_train.toml`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/contests/ahc-061/configs/ppo_train.toml:1)

## 参照した SimbaV2 側

- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py>
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_network.py>
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_update.py>
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/wrappers/normalization.py>
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/configs/agent/simbaV2.yaml>
