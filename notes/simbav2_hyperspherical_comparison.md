# SimbaV2 の hyperspherical 系実装との比較メモ

対象:
- ローカル実装: [`src/ahcrl/nn/blocks.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:9)
- ローカル利用箇所: [`src/ahcrl/contests/ahc061/model.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/model.py:8), [`src/ahcrl/contests/ahc061/train_ppo.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:67)
- 参照実装: <https://github.com/DAVIAN-Robotics/SimbaV2>

このメモは、`spherical` / `l2 norm` / `scale` 周りを SimbaV2 に寄せる際の差分整理を目的にしている。

## 結論

ローカル実装は、SimbaV2 の「発想」をかなり取り込んでいるが、**ブロック構造そのものは一致していない**。

特に一致していない点は次の3つ。

1. `ShiftL2Norm` は SimbaV2 の embedder の前半だけを切り出したもので、後段の `Linear + Scaler + l2 norm` がない。
2. `SphericalConvNeXtBlock` は、SimbaV2 の `HyperLERPBlock` ではない。`LERP` の学習係数 `alpha` と最終 `l2 norm` の構造が違う。
3. 重みの l2 投影は SimbaV2 では更新後に毎回行う設計だが、ローカルでは `weight_projection=True` のときだけで、しかもデフォルトは無効。

つまり、現在のコードは「SimbaV2 の要素を参考にした別設計」と見るのが正確で、SimbaV2 の再現実装とは言いにくい。

## SimbaV2 側の要点

参照実装の hyperspherical 系は、`scale_rl/agents/simbaV2/simbaV2_layer.py` と `scale_rl/agents/simbaV2/simbaV2_update.py` にまとまっている。

代表的な流れは以下。

1. 観測を `Shift + l2 norm` で埋め込む
2. `Linear + Scaler` で hypersphere に戻す
3. `MLP + l2 norm` を繰り返す
4. 各 block は `residual + alpha * (transform - residual)` の LERP を使う
5. 学習更新後に `hyper_dense` の重みを l2 正規化する

該当箇所:
- [`HyperEmbedder`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L68-L86)
- [`HyperMLP`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L46-L65)
- [`HyperLERPBlock`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L89-L117)
- [`l2normalize_network`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_update.py#L24-L46)

ドキュメント上でも、SimbaV2 は次のように説明されている。

- shift を付加して l2 正規化する
- `Linear + Scaler` で埋め込む
- `MLP + l2 norm` を block として積む
- `LERP + l2 norm` で block をまとめる

該当説明:
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/docs/index.html#L106-L125>

## ローカル実装との対応表

### 1. `Scaler`

ローカルの [`Scaler`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:147) は、発想としては SimbaV2 とかなり近い。

SimbaV2 側:
- `scaler` を `scale` で初期化
- forward では `init / scale` を掛ける

ローカル側:
- `scaler = nn.Parameter(full((dim,), scale))`
- `forward_scaler = init / scale`
- `x * scaler * forward_scaler`

これは実質同じ設計に見てよい。

ただし、ローカル版には `dim` の形状チェックや `scale != 0` のバリデーションが追加されていて、実装としては少し堅い。

### 2. `l2_normalize`

ローカルの [`l2_normalize`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:9) は、SimbaV2 の `l2normalize` と同じ役割を持つ。

SimbaV2 側:
- `jnp.linalg.norm(..., keepdims=True)`
- `x / max(norm, EPS)`

ローカル側:
- `x.float().pow(2).sum(...).clamp_min(eps**2).sqrt()`
- `x / norm`

違いはほぼ実装詳細だけで、機能面のズレは小さい。

注意点として、ローカル版は `eps` を `norm` の下限として二乗で扱っている。数値的には自然だが、SimbaV2 の `max(l2norm, EPS)` と同等の狙いであることは明記しておいた方がよい。

### 3. `ShiftL2Norm`

ローカルの [`ShiftL2Norm`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:117) は、SimbaV2 の `HyperEmbedder` の前半部分に対応する。

SimbaV2 の embedder は:

1. 入力に `c_shift` を1次元足す
2. `l2normalize`
3. `HyperDense`
4. `Scaler`
5. `l2normalize`

ローカルの `ShiftL2Norm` は 1 と 2 だけをやる。

なので、`ShiftL2Norm` 単体では SimbaV2 の embedder を再現していない。  
もし「SimbaV2 風の埋め込み」を狙うなら、少なくとも `ShiftL2Norm -> LinearScaler(normalize_output=True)` か、1つの専用モジュールで `Shift -> Linear -> Scaler -> l2` まで持つ必要がある。

根拠:
- [`HyperEmbedder`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L68-L86)
- [`ShiftL2Norm`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:117)

### 4. `LinearScaler`

ローカルの [`LinearScaler`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:169) は、SimbaV2 の `HyperDense + Scaler + l2normalize` を一般化したものに近い。

ただし厳密には次が違う。

- SimbaV2 は `nn.Dense` の kernel 初期化に orthogonal を使うだけでなく、`l2normalize_network` により更新後も重みを投影する
- ローカルは `nn.init.orthogonal_` の後に `project_weight_to_unit_norm_` を一度だけ呼ぶ
- さらに `normalize_output=True` のときだけ出力を l2 正規化する

つまりローカル `LinearScaler` は「初期化時に近い状態」にはできるが、「更新後も hypersphere を保つ」ところまでは自動ではない。

### 5. `SphericalConvNeXtBlock`

ここが一番大きい差分。

ローカルの [`SphericalConvNeXtBlock`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:232) は、

- `depthwise conv`
- `pre_norm`
- `pointwise MLP`
- `layer_scale`
- `post_norm`

という ConvNeXt 派生の構造になっている。

一方 SimbaV2 の block は `HyperLERPBlock` で、

- `HyperMLP`
- `residual + alpha * (x - residual)`
- `l2normalize`

の構造を取る。

つまりローカルの block は以下が異なる。

- `alpha` による learnable interpolation がない
- `depthwise conv` が入っている
- `LayerNorm` 相当の処理がない
- `MLP` の中の `eps` によるゼロベクトル回避も構造的には別

要するに、名前に `Spherical` が付いているが、SimbaV2 の `HyperLERPBlock` の直訳ではない。

根拠:
- [`HyperLERPBlock`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py#L89-L117)
- [`SphericalConvNeXtBlock`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:232)

### 6. 重み投影

SimbaV2 の更新規則では、`hyper_dense` に対応する重みを毎回 l2 正規化している。

ローカルでは [`project_hyperspherical_weights_`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:220) があるが、`train_ppo.py` では `args.weight_projection` が有効なときだけ呼ばれる。

さらに、対象は `Linear/Conv1d/Conv2d/Conv3d` 全体であり、SimbaV2 の「特定の hyper_dense のみ」より広い。

実運用上の差分はここが大きい。

- SimbaV2 はデフォルトで正規化が走る
- ローカルは明示フラグを立てないと走らない

根拠:
- [`l2normalize_network`](https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_update.py#L37-L46)
- [`project_hyperspherical_weights_`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:220)
- [`train_ppo.py` の投影呼び出し](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:599)

## 実装上の含意

この差分を放置したまま「SimbaV2 風」と呼ぶと、後で比較実験したときに何が効いたのか分からなくなる。

特に注意が必要なのは次。

1. `spherical_convnext` は SimbaV2 の block ではないので、性能差が出ても「Simbav2 の再現に失敗した」のか「別構造が効いた」のか切り分けにくい
2. `weight_projection=False` のままだと、SimbaV2 の core assumption の1つである「更新後の重み l2 制約」が入っていない
3. `ShiftL2Norm` 単体は、観測の magnitude を残す補助としては有用だが、SimbaV2 の embedder 全体とは別

## 実装を寄せるなら

SimbaV2 により寄せるなら、優先順位は次が妥当。

1. `SphericalConvNeXtBlock` を `HyperLERPBlock` に置き換える、あるいは別クラスとして追加する
2. `ShiftL2Norm` と `LinearScaler(normalize_output=True)` を組み合わせた embedder を作る
3. 学習更新後の重み投影をデフォルト有効にする
4. `alpha` 相当の learnable interpolation と初期値を導入する
5. `model_block_type` の命名を、SimbaV2 再現と別設計であることが分かるように整理する

## 参照したローカル箇所

- [`src/ahcrl/nn/blocks.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/nn/blocks.py:9)
- [`src/ahcrl/contests/ahc061/model.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/model.py:8)
- [`src/ahcrl/contests/ahc061/train_ppo.py`](/home/harui/CompetitiveProgramming/ahc/ahcrl-meta/src/ahcrl/contests/ahc061/train_ppo.py:67)

## 参照した SimbaV2 側

- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_layer.py>
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/scale_rl/agents/simbaV2/simbaV2_update.py>
- <https://github.com/DAVIAN-Robotics/SimbaV2/blob/main/docs/index.html>

