"""AHC068用Actor-Criticモデルの骨格。

このファイルでは意図的に方策全体の実装までは行わない。
モデルとPPOのインターフェースを明確にすることが目的である。

* ``ActorCritic.forward`` はアクションをサンプリングしない。
* 最初のforwardパスは、最初の2つの因子のロジットを生成する。
* ``endpoint_logits`` は、呼び出し側が方向とアンカーセルを選択した後、
  最後の因子のロジットを生成する。
* アクション空間固有の合法性マスクと因子分解サンプリングは、
  ニューラルネットワークではなくPPO/方策層に属する。

AHC068では、最後の因子を長方形の右下セルで表す。アクション空間の実装は
``(direction, anchor, endpoint)`` を出力タプル ``(V/H, r, c, h, w)`` に変換し、
合法な長方形を表さないエンドポイントを拒否しなければならない。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from ahcrl.contests.ahc068.encoder import BOARD_SIZE, NUM_PLANES
from ahcrl.nn.blocks import ConvNeXtBlock, ResidualBlock
from ahcrl.nn.components import make_group_norm

NUM_DIRECTIONS = 2
"""ここでは0を縦、1を横とする。"""


class FactorizedActionLogits(NamedTuple):
    """因子分解したアクション決定を開始するために必要なモデル出力。

    ``anchor_logits`` は方向ごとに1枚の空間マップを持つ。選択された方向を
    使ってPPO方策層でマップの1枚を選択する。

    ``features`` 自体はロジットではない。トランクを2回実行せずに、呼び出し側が
    次の条件付き分布を ``endpoint_logits`` に問い合わせられるよう返している。

    TODO(student): ここでトランクの特徴量を公開するAPIが適切か決める。
    別の ``encode`` メソッドやトランクの再計算も合理的な選択肢であり、
    メモリと速度のトレードオフが異なる。
    """

    direction_logits: torch.Tensor  # 形状 [B, 2]
    anchor_logits: torch.Tensor  # 形状 [B, 2, N, N]
    features: torch.Tensor  # 形状 [B, C, N, N]


class ActorCritic(nn.Module):
    """因子分解したアクターとスカラー値関数を持つ共有CNN。

    この教材用骨格で使う方策の因子分解は次のとおりである。

    ``p(direction, anchor, endpoint | state)``
    ``= p(direction | state)``
    ``* p(anchor | state, direction)``
    ``* p(endpoint | state, direction, anchor)``

    ネットワークはこれらの分布のロジットだけを生成する。``softmax``、
    ``multinomial``、``Categorical.sample`` は呼び出さない。
    """

    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
        block_type: str = "convnext",
        head_channels: int = 8,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if channels <= 0 or blocks <= 0 or head_channels <= 0:
            raise ValueError("channels, blocks and head_channels must be positive")
        if block_type not in ("convnext", "residual"):
            raise ValueError(f"unknown block_type: {block_type}")

        block_class = ConvNeXtBlock if block_type == "convnext" else ResidualBlock
        self.channels = channels

        # TODO(student): 観測プレーンを確定した後で入力カーネルサイズを選ぶ。
        # 1x1埋め込みにより、この実験の焦点を方策インターフェースに保てる。
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, padding=0, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            *[block_class(channels) for _ in range(blocks)],
        )

        # プーリングは方向ヘッドと値ヘッドにだけ適用する。アンカーと
        # エンドポイントの出力は盤面上のセルの選択なので、空間情報を保持する。
        pooled_channels = channels * 2
        self.direction_head = nn.Sequential(
            nn.Linear(pooled_channels, channels),
            nn.ReLU(inplace=True),
            # すべての方向で共有するバイアスはsoftmaxに対して不変である。
            # bias=Falseにして、その事実が分かるようにしている。
            nn.Linear(channels, NUM_DIRECTIONS, bias=False),
        )

        # ``anchor_logits[:, d, r, c]`` は方向dを選択したときの左上セル(r, c)
        # のスコアである。
        self.anchor_head = nn.Sequential(
            nn.Conv2d(channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, NUM_DIRECTIONS, kernel_size=1, bias=False),
        )

        # エンドポイントヘッドは方向とアンカーを選択した後に呼び出す。
        # 2枚の方向one-hotプレーンと1枚のアンカープレーンを受け取り、
        # 右下セルの候補ごとに1つのロジットを出力する。
        #
        # TODO(student): この明示的な条件付けを、アクション埋め込み、FiLM、
        # または高さ/幅を直接予測するヘッドと比較する。
        endpoint_input_channels = channels + NUM_DIRECTIONS + 1
        self.endpoint_head = nn.Sequential(
            nn.Conv2d(endpoint_input_channels, head_channels, kernel_size=1),
            ConvNeXtBlock(head_channels),
            nn.Conv2d(head_channels, 1, kernel_size=1, bias=False),
        )

        self.value_head = nn.Sequential(
            nn.Linear(pooled_channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[FactorizedActionLogits, torch.Tensor]:
        """因子分解したロジットと値の推定値をそのまま返す。

        Args:
            x: 形状 ``[B, NUM_PLANES, N, N]`` のエンコード済み観測。

        Returns:
            ``(action_logits, value)``。``action_logits`` は方向ロジット、
            方向ごとのアンカーロジット、``endpoint_logits`` に必要な潜在特徴量を
            含む。``value`` の形状は ``[B]``。

        サンプリングと合法性マスクは意図的に含めていない。PPOではロールアウト
        コードが各因子をサンプリングし、それぞれの対数確率を加算する。
        """
        self._validate_observation(x)
        features = self.trunk(x)
        self._validate_features(features, x.shape[0])

        pooled = torch.cat(
            [features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))],
            dim=1,
        )
        direction_logits = self.direction_head(pooled)
        anchor_logits = self.anchor_head(features)
        value = self.value_head(pooled).squeeze(-1)

        assert direction_logits.shape == (x.shape[0], NUM_DIRECTIONS)
        assert anchor_logits.shape == (
            x.shape[0],
            NUM_DIRECTIONS,
            BOARD_SIZE,
            BOARD_SIZE,
        )
        assert value.shape == (x.shape[0],)

        return FactorizedActionLogits(direction_logits, anchor_logits, features), value

    def endpoint_logits(
        self,
        features: torch.Tensor,
        direction: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """選択済みのプレフィックスを条件とするエンドポイントロジットを返す。

        Args:
            features: ``forward`` のトランク出力。形状は ``[B, C, N, N]``。
            direction: 形状 ``[B]`` のLongテンソル。0はV、1はH。
            anchor: 平坦化したセルのインデックス ``r * N + c`` を含む、
                形状 ``[B]`` のLongテンソル。

        Returns:
            形状 ``[B, N, N]`` のエンドポイントロジット。PPO方策はこのテンソルを
            平坦化し、エンドポイントの合法性マスクを適用しなければならない。

        ニューラルネットワークはアクション空間の完全な意味規則を知らない。
        例えば、縦長方形に偶数の高さが必要なことや、内部の辺に壁があっては
        ならないことを知らない。これらの規則は ``action_space.py`` が生成する
        マスクに属する。
        """
        self._validate_features(features, features.shape[0])
        batch_size, _, height, width = features.shape
        if direction.shape != (batch_size,) or anchor.shape != (batch_size,):
            raise ValueError("direction and anchor must both have shape [B]")
        if height != BOARD_SIZE or width != BOARD_SIZE:
            raise ValueError("AHC068 features must have spatial shape [N, N]")

        direction = direction.long()
        anchor = anchor.long()
        if torch.any((direction < 0) | (direction >= NUM_DIRECTIONS)):
            raise ValueError("direction contains an invalid factor")
        if torch.any((anchor < 0) | (anchor >= BOARD_SIZE * BOARD_SIZE)):
            raise ValueError("anchor contains an invalid cell index")

        # 方向のone-hotプレーンは選択済みプレフィックスを盤面全体に広げる。
        # アンカープレーンはバッチ内の各要素について1セルだけを示す。
        direction_planes = F.one_hot(direction, num_classes=NUM_DIRECTIONS)
        direction_planes = direction_planes.to(device=features.device, dtype=features.dtype)
        direction_planes = direction_planes[:, :, None, None].expand(-1, -1, height, width)

        anchor_plane = torch.zeros(
            (batch_size, height * width),
            device=features.device,
            dtype=features.dtype,
        )
        anchor_plane.scatter_(1, anchor[:, None], 1.0)
        anchor_plane = anchor_plane.view(batch_size, 1, height, width)

        # 重要TODO(student): アンカープレーンを連結するだけでは、必ずしも
        # 大域的な条件付けにはならない。浅い局所CNNでは、アンカーから遠い
        # エンドポイントがアンカーに関する有用な情報を受け取れない可能性がある。
        # 次の実験では、``anchor`` の特徴量を取り出してすべてのセルに広げるか、
        # アクション埋め込み/Attention機構を使うことを検討できる。
        conditioned = torch.cat((features, direction_planes, anchor_plane), dim=1)
        logits = self.endpoint_head(conditioned).squeeze(1)
        assert logits.shape == (batch_size, BOARD_SIZE, BOARD_SIZE)
        return logits

    @staticmethod
    def select_anchor_logits(
        anchor_logits: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """選択された各方向に対応するアンカーマップを選ぶ。

        これはサンプリングではなく、テンソルだけの演算である。返されるテンソル
        の形状は ``[B, N, N]`` で、PPO層のマスク付きカテゴリ分布に渡せる。
        """
        if anchor_logits.ndim != 4:
            raise ValueError("anchor_logits must have shape [B, 2, N, N]")
        if anchor_logits.shape[1] != NUM_DIRECTIONS:
            raise ValueError("anchor_logits has an unexpected direction axis")
        if direction.shape != (anchor_logits.shape[0],):
            raise ValueError("direction must have shape [B]")
        selected = anchor_logits.gather(
            1,
            direction.long().view(-1, 1, 1, 1).expand(-1, 1, BOARD_SIZE, BOARD_SIZE),
        )
        return selected.squeeze(1)

    def _validate_observation(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        if x.shape[1] != self.trunk[0].in_channels:
            raise ValueError(
                f"expected {self.trunk[0].in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[2:] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError("AHC068 observations must have spatial shape [N, N]")

    def _validate_features(self, features: torch.Tensor, batch_size: int) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, C, N, N]")
        if features.shape[0] != batch_size:
            raise ValueError("features and batch_size disagree")
        if features.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} feature channels, got {features.shape[1]}")
        if features.shape[2:] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError("features must have spatial shape [N, N]")


class FactorizedPolicy:
    """PPO側の方策アダプターの骨格。

    このクラスは意図的に未完成である。``action_space.py`` の準備ができたときに
    ``ActorCritic`` の外部に存在すべきシグネチャを示している。この層が返す
    アクションは、複数の因子を通じてサンプリングされるが、最終的には単一の
    平坦化されたアクションIDになるべきである。
    """

    def sample(
        self,
        model: ActorCritic,
        observations: torch.Tensor,
        action_masks: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """アクションをサンプリングし、``(action_id, log_prob, entropy)`` を返す。

        アクション空間APIが設計されるまで、``action_masks`` は ``object`` のままに
        している。おそらく各因子のマスクを1つずつ含み、サンプリング済みの
        プレフィックスに依存する可能性がある。
        """
        del model, observations, action_masks
        # TODO(student):
        # 1. model(observations)を呼び出す。
        # 2. 方向ロジットをマスクして方向をサンプリングする。
        # 3. アンカーロジットを選択・マスクしてアンカーをサンプリングする。
        # 4. model.endpoint_logits(...)を呼び出す。
        # 5. マスクしてエンドポイントをサンプリングする。
        # 6. 3つの対数確率を加算する。
        #
        # これらの手順をActorCritic.forwardに入れてはならない。
        raise NotImplementedError("TODO: implement factorized PPO sampling")

    def evaluate_actions(
        self,
        model: ActorCritic,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """保存済みアクションを ``(log_prob, entropy, value)`` として評価する。

        ``actions`` にはロールアウトで保存した平坦化アクションIDを含める。
        アクション空間層は、3つのカテゴリ分布の対数確率を合計する前に、各IDを
        方向、アンカー、エンドポイントへデコードしなければならない。
        """
        del model, observations, actions, action_masks
        # TODO(student): ``sample`` のサンプリングしない版を実装する。
        # 特に、ロールアウトとPPOのミニバッチ更新でまったく同じマスクと
        # 因子の順序を使うことに注意する。
        raise NotImplementedError("TODO: implement factorized PPO evaluation")
