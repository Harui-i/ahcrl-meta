"""Contest-specific Actor-Critic model."""

# TODO: Reuse blocks from ahcrl.nn and implement the policy/value heads.

import torch
from torch import nn
from ahcrl.contests.ahc068.encoder import BOARD_SIZE, NUM_PLANES

from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm

"""
Dear Copilotさんへ: plz read problem_en.md for details and context.

ここで特徴量決めてから、諸々の設計が決まり、それらをencoder.pyに反映していくという流れでやっていきましょうね。

まあ入力特徴量はCNNか。 planeをどうするかとかは後で決められるので一旦おいとく。
出力ヘッドをどうするか、という問題がある

行動の決め方は、
1: (縦の長方形にするのか vs 横の長方形にするのか)という決定と、
2: 長方形の範囲はどこにしますか？ という２つ

こういうのってどうやって決めるのが良いんだろう。ナイーブに考えると縦/横判定ヘッドに縦横を出させて、
output = rectangle_head(feature, tate_or_yoko) みたいな感じでtate/yoko条件付けで長方形の範囲を出す感じになるよな。
まあそれで一旦良いと思う。
で、rectangle_headをどう出力させるんですか?という問題があるよね
例えばここも自己回帰的に、左上出力ヘッドと、その条件付のもとで右下出力ヘッドを出す、みたいな感じでやるのが良いのか？
まあええか

モデル再実装したくないし、表現を再利用してくれると嬉しいから,
左上出力ヘッドと右下出力ヘッドは同じパラメタで、条件付けさせるのがいいかな.
"""


class ActorCritic(nn.Module):
    def __init__(
            self, 
            in_channels: int = NUM_PLANES,
            channels: int = 64,
            blocks: int = 4,
    ):
        super().__init__()

        if channels <= 0 or blocks <= 0:
            raise ValueError("channels and blocks must be positive")

        # TODO : normalize処理をどこかでかます。一旦面倒なのでpostpone
        self.trunk = nn.Sequential(
            # embedding
            # ここ3x3にする実装あるけどどうなんだ 
            nn.Conv2d(in_channels, channels, kernel_size=1, padding=0, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            *[ConvNeXtBlock(channels) for _ in range(blocks)],
        )
        # これで (N, channels, BOARD_SIZE, BOARD_SIZE) になる
        """ で、どうするかって話
tate_or_yoko head: 最終的にはbinary classification.
ただどういう構造にしますかという話。テキトーにavgpoolしたりするのも考えられるけど、
→いや、tate_or_yokoの値をマスごとに出すのもありかもしれない。

ん？分割する必要なくね？ (N,2, BOARD_SIZE, BOARD_SIZE)の出力で、各マス2つlogitださせて、
１つ目のlogitはtateとしてそこを左上にするというアクションのlogit,2つ目のlogitはyokoとしてそこを左上にするというアクションのlogitとして使うのが良いかもしれない。

そしたら分割が一個へるよね。Q:じゃあ右下をどう作りますかという話なんですが、…
A: 
        """
        # out[n][i][j][y] := マス(i,j)を(左上 if y == 0 else 右下)にするというlogit
        self.policy_tateyoko_pointer = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=True),
        ) # output shape: (N, 1, BOARD_SIZE, BOARD_SIZE)

        """
左上→右下の順で決めることにしちゃう。
そして、channelは2増やす.
新しいchannel(1)はtate/yokoのone-hot uniform planeとして使うことにする.
新しいchannel(2)は左上として選ばれたマスを1にするplaneとして使うことにする.
これで左上でも右下でも同じパラメタで出力させられる。
        """
        self.policy_migisita_pointer = nn.Sequential(
            nn.Conv2d(channels + 2, 8, kernel_size=1, padding=0, bias=True),
            ConvNeXtBlock(8),
            # logitをマスごとに出す. biasはいらない…よね?
            nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False),
        ) # output shape: (N, 1, BOARD_SIZE, BOARD_SIZE)
        
        self.value = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1, padding=0, bias=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        h: torch.Tensor = self.trunk(x)
        assert h.shape == (x.shape[0], 64, BOARD_SIZE, BOARD_SIZE)

        value: torch.Tensor = self.value(h)
        tate_or_yoko: torch.Tensor = self.policy_tateyoko_pointer(h)

        assert value.shape == (x.shape[0], 1)
        assert tate_or_yoko.shape == (x.shape[0], 2, BOARD_SIZE, BOARD_SIZE)

        # (この時点で、どのマスを左上にするか、そしてそこで縦にするか横にするかが決まっている)
        # 左上 * 縦or横　のサンプリングはどうやるのがいいんだ
        # 候補としてはsoftmaxかargmax. argmaxはgreedyすぎるのでsoftmaxでサンプリングするのが良いかもしれない。
        # forwardの中でサンプルしちゃっていいのか？
        # まあいいか。とりあえずsoftmaxでサンプルすることにする。
        # (N, 2, BOARD_SIZE, BOARD_SIZE)のlogitにしてるのだるいな
        # (N, BOARD_SIZE*BOARD_SIZE*2) のlogitにすればsoftmaxでサンプルできるな。まあいいか。
        tate_or_yoko_flat: torch.Tensor = tate_or_yoko.flatten(start_dim=1) # shape: (N, BOARD_SIZE*BOARD_SIZE*2)
        tate_or_yoko_probs: torch.Tensor = torch.softmax(tate_or_yoko_flat, dim=1)
        tate_or_yoko_sample: torch.Tensor = torch.multinomial(tate_or_yoko_probs, num_samples=1) # shape: (N, 1)

        # ここで、左上のマスと縦横の情報を取り出す
        tate_or_yoko_sample_unflat: torch.Tensor = tate_or_yoko_sample.view(-1, 1, 1) # shape: (N, 1, 1)
        tate_or_yoko_sample_unflat = tate_or_yoko_sample_unflat % (BOARD_SIZE * BOARD_SIZE * 2) # shape: (N, 1, 1)
        tate_or_yoko_sample_unflat = tate_or_yoko_sample_unflat // 2 # shape: (N, 1, 1)
        tate_or_yoko_sample_unflat = tate_or_yoko_sample_unflat.view(-1) # shape: (N,)
        # これで、左上のマスのindexが取れる. そして縦横の情報はtate_or_yoko_sample % 2で取れる.
