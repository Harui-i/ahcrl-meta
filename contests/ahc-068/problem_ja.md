# A - Rectangle Swap

`N \times N` マスの盤面がある。
左上のマスの座標を `(0,0)` とし、そこから下方向に `i` マス、右方向に `j` マス進んだ先のマスの座標を `(i,j)` とする。
盤面の外周は壁で囲まれており、隣接するマスの間にも壁が存在する場合がある。
すべてのマスは、壁のない隣接マスへの上下左右の移動を繰り返すことで、互いに到達可能であることが保証されている。
左上が `(r,c)`、高さが `h>0`、幅が `w>0` の長方形を `(r,c,h,w)` と書き、以下のマスの集合として定義する。

\{(r+x,c+y) \mid 0 \leq x < h,\ 0 \leq y < w\}

各マスには、`0,1,\dots,N^2-1` の番号が書かれたカードが 1 枚ずつ置かれている。
初期状態におけるカードの配置が与えられるので、以下の操作を繰り返し、最終的に各マス `(i,j)` に番号 `iN+j` のカードが置かれている状態にしたい。
各操作では、まず向きとして縦または横を選び、さらに以下の条件を満たす長方形 `R=(r,c,h,w)` を選ぶ。

- 縦を選んだ場合、高さ `h` は偶数でなければならない。
- 横を選んだ場合、幅 `w` は偶数でなければならない。
- `R` のすべてのマスは盤面内に含まれる。
- `R` に含まれる任意の隣接する 2 マスの間に壁が存在しない。

次に、選んだ長方形に含まれるカードを以下のように入れ替える。

- 縦を選んだ場合、すべての整数 `x,y` であって、`0\leq x<h/2` かつ `0\leq y<w` を満たすものに対し、マス `(r+x,c+y)` のカードとマス `(r+h/2+x,c+y)` のカードを入れ替える。
- 横を選んだ場合、すべての整数 `x,y` であって、`0\leq x<h` かつ `0\leq y<w/2` を満たすものに対し、マス `(r+x,c+y)` のカードとマス `(r+x,c+w/2+y)` のカードを入れ替える。

壁の配置は操作によって変化しない。

操作は最大 `10^5` 回行うことができる。
できるだけ少ない操作回数で目的の配置を達成せよ。

### 得点

出力した操作回数を `T` とする。
すべての操作を行った後の盤面において、目的のカードが置かれていないマスの個数を `E` とする。
このとき、以下の得点が得られる。

- `E=0` の場合、`N^2 + \mathrm{round}\left(10^6 \times \log_2 \frac{10^5}{T}\right)`
- `E>0` の場合、`N^2-E`

合計で 150 個のテストケースがあり、各テストケースの得点の合計が提出の得点となる。
一つ以上のテストケースで不正な出力や制限時間超過をした場合、提出全体の判定が WA や TLE となる。
コンテスト時間中に得た最高得点で最終順位が決定され、コンテスト終了後のシステムテストは行われない。
同じ得点を複数の参加者が得た場合、提出時刻に関わらず同じ順位となる。

---

### 入力

入力は以下の形式で標準入力から与えられる。

```
`N`
`a_{0,0}` `\cdots` `a_{0,N-1}`
`\vdots`
`a_{N-1,0}` `\cdots` `a_{N-1,N-1}`
`V_0`
`\vdots`
`V_{N-1}`
`H_0`
`\vdots`
`H_{N-2}`

```

- すべてのテストケースにおいて、盤面サイズ `N` は `20` に固定されている。
- `a_{i,j}` は、初期状態においてマス `(i,j)` に置かれているカードの番号を表す。
- `a_{i,j}` 全体は、`0,1,\dots,N^2-1` の順列である。
- 少なくとも 1 つのマス `(i,j)` が `a_{i,j}\neq iN+j` を満たす。
- 各 `V_i` は、0 と 1 からなる長さ `N-1` の文字列であり、その `j` 文字目は、マス `(i,j)` とマス `(i,j+1)` の間に壁がある (1) かない (0) かを表す。
- 各 `H_i` は、0 と 1 からなる長さ `N` の文字列であり、その `j` 文字目は、マス `(i,j)` とマス `(i+1,j)` の間に壁がある (1) かない (0) かを表す。
- すべてのマスは、壁のない隣接マスへの上下左右の移動を繰り返すことで、互いに到達可能である。

### 出力

操作回数を `T\ (0\leq T\leq 10^5)` とする。
`t` 回目の操作で選んだ向き `d_t` を、縦の場合は V、横の場合は H として、アルファベット 1 文字で表す。
また、`t` 回目の操作で選んだ長方形を `R_t=(r_t,c_t,h_t,w_t)` とする。
このとき、以下の形式で標準出力に出力せよ。

```
`d_0` `r_0` `c_0` `h_0` `w_0`
`\vdots`
`d_{T-1}` `r_{T-1}` `c_{T-1}` `h_{T-1}` `w_{T-1}`

```

例を見る

### 入力生成方法

`N=20` とする。

`L` 以上 `U` 以下の整数を一様ランダムに生成する関数を `\mathrm{rand}(L,U)` とする。

### 壁の生成

マスの角に位置する点を「頂点」と呼ぶ。

`W=\mathrm{rand}(0,N-1)` を生成する。
マス間に壁がなく、盤面の外周のみが壁で囲まれた状態から開始し、以下の操作を `W` 回繰り返す。
いずれの壁にも接していない頂点の中から、一様ランダムに 1 つを選ぶ。
次に、上下左右の 4 方向の中から、一様ランダムに 1 つを選ぶ。
選んだ頂点から選んだ方向に向かって、盤面の外周または既存の壁に接するまで壁を伸ばす。

### 初期盤面の生成

初期盤面におけるカードの配置は、`0,1,\dots,N^2-1` のすべての順列の中から一様ランダムに生成される。
すべての `(i,j)` に対して `a_{i,j}=iN+j` である場合は、生成し直す。

### ツール(入力ジェネレータ・ビジュアライザ)

- Web 版: ローカル版より高性能であり、アニメーション表示が可能である。
ローカル版: 使用するにはRust 言語のコンパイル環境を用意せよ。
- Windows 用のコンパイル済みバイナリ: Rust 言語の環境構築が面倒な場合は、代わりにこちらを利用せよ。

コンテスト期間中に、ビジュアライズ結果の共有や、解法・考察に関する言及は禁止されているため注意せよ。

### 生成AIの利用に関して

現行の AtCoder Heuristic Contest 生成AI利用ルール－20250616版 では、AIエージェントに解答プログラムを実行させ、その実行結果に基づく改善を自動的に繰り返させることを禁止しています。

一方、最新の生成AIでは、ChatGPTのような対話型サービスと、CodexのようなAIエージェントとの区別が曖昧になっています。 対話型サービスであっても、利用者が明示的に指示していない場合に、内部でテストケースの生成、解答プログラムの実行、および実行結果に基づく改善を 自動的に繰り返すことがあります。

一部の参加者は、このような挙動を防ぐため、生成AIに対して禁止事項を明示しています。参加者間の公平性を確保し、現行ルールの趣旨を徹底するため、本コンテストで生成AIを利用する場合は、以下の指示文を各チャットの冒頭に入力するか、各生成AIツールが提供するカスタム指示、プロジェクト指示、または自動的に読み込む指示ファイル （例：AGENTS.md、CLAUDE.mdなど）に設定しなければなりません。

```
I am currently participating in an AtCoder Heuristic Contest, and I will use this generative AI to assist in developing my solution.

When using this generative AI, the "AtCoder Heuristic Contest Generative AI Usage Rules - Version 20250616" apply.

https://info.atcoder.jp/entry/ahc-llm-rules-en

Most importantly, after running the solution program, you must not modify or improve the solution, its approach, or its code based on the execution results unless the user gives a new explicit instruction to do so.

You may run the solution program and report its execution results, logs, scores, or other observations. After reporting them, you must stop and wait for a new instruction from the user before making any improvement based on those results.

Here, "solution program" refers to any program created or being created for the purpose of solving this contest problem, regardless of whether it was created by the user or by generative AI, and regardless of whether it is still in progress or already complete.
```

```
I am currently participating in an AtCoder Heuristic Contest, and I will use this generative AI to assist in developing my solution.

When using this generative AI, the "AtCoder Heuristic Contest Generative AI Usage Rules - Version 20250616" apply.

https://info.atcoder.jp/entry/ahc-llm-rules-en

Most importantly, after running the solution program, you must not modify or improve the solution, its approach, or its code based on the execution results unless the user gives a new explicit instruction to do so.

You may run the solution program and report its execution results, logs, scores, or other observations. After reporting them, you must stop and wait for a new instruction from the user before making any improvement based on those results.

Here, "solution program" refers to any program created or being created for the purpose of solving this contest problem, regardless of whether it was created by the user or by generative AI, and regardless of whether it is still in progress or already complete.

```
