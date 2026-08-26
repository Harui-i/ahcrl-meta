# A - Rectangle Swap

### Problem Statement

There is an `N \times N` grid.
Let the coordinates of the top-left cell be `(0,0)`. The coordinates of the cell located `i` cells downward and `j` cells to the right from it are `(i,j)`.
The outer boundary of the grid is surrounded by walls, and walls may also exist between adjacent cells.
It is guaranteed that every cell is reachable from every other cell by repeatedly moving up, down, left, or right between adjacent cells without a wall between them.
A rectangle whose top-left cell is `(r,c)`, height is `h>0`, and width is `w>0` is denoted by `(r,c,h,w)` and defined as the following set of cells:

\{(r+x,c+y) \mid 0 \leq x < h,\ 0 \leq y < w\}

Each cell contains exactly one card labeled with an integer from `0,1,\dots,N^2-1`.
Given the initial arrangement of the cards, repeat the following operation to obtain a state in which each cell `(i,j)` contains the card labeled `iN+j`.
In each operation, first choose either the vertical or horizontal direction, and then choose a rectangle `R=(r,c,h,w)` satisfying the following conditions:

- If the vertical direction is chosen, the height `h` must be even.
- If the horizontal direction is chosen, the width `w` must be even.
- Every cell in `R` is contained within the grid.
- There is no wall between any pair of adjacent cells contained in `R`.

Then, swap the cards contained in the chosen rectangle as follows:

- If the vertical direction is chosen, for every pair of integers `x,y` satisfying `0\leq x<h/2` and `0\leq y<w`, swap the card in cell `(r+x,c+y)` with the card in cell `(r+h/2+x,c+y)`.
- If the horizontal direction is chosen, for every pair of integers `x,y` satisfying `0\leq x<h` and `0\leq y<w/2`, swap the card in cell `(r+x,c+y)` with the card in cell `(r+x,c+w/2+y)`.

The wall configuration does not change as a result of the operations.

You may perform at most `10^5` operations.
Achieve the target arrangement using as few operations as possible.

### Scoring

Let `T` be the number of operations in your output.
After performing all operations, let `E` be the number of cells that do not contain their target card.
Your score is calculated as follows:

- If `E=0`, `N^2 + \mathrm{round}\left(10^6 \times \log_2 \frac{10^5}{T}\right)`
- If `E>0`, `N^2-E`

There are `150` test cases, and the score of a submission is the total score for each test case.
If your submission produces an illegal output or exceeds the time limit for some test cases, the submission itself will be judged as WA or TLE , and the score of the submission will be zero.
The highest score obtained during the contest will determine the final ranking, and there will be no system test after the contest.
If more than one participant gets the same score, they will be ranked in the same place regardless of the submission time.

---

### Input

The input is given from Standard Input in the following format:

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

- The grid size `N` is fixed to `20` in all test cases.
- `a_{i,j}` represents the number of the card initially placed in cell `(i,j)`.
- The values `a_{i,j}` form a permutation of `0,1,\dots,N^2-1`.
- There is at least one cell `(i,j)` satisfying `a_{i,j}\neq iN+j`.
- Each `V_i` is a string of length `N-1` consisting of 0 and 1. Its `j`-th character is 1 if there is a wall between cells `(i,j)` and `(i,j+1)`, and 0 otherwise.
- Each `H_i` is a string of length `N` consisting of 0 and 1. Its `j`-th character is 1 if there is a wall between cells `(i,j)` and `(i+1,j)`, and 0 otherwise.
- Every cell is reachable from every other cell by repeatedly moving up, down, left, or right between adjacent cells without a wall between them.

### Output

Let `T\ (0\leq T\leq 10^5)` be the number of operations.
Represent the direction `d_t` chosen in the `t`-th operation by a single letter: V for vertical and H for horizontal.
Let the rectangle chosen in the `t`-th operation be `R_t=(r_t,c_t,h_t,w_t)`.
Output the operations to Standard Output in the following format:

```
`d_0` `r_0` `c_0` `h_0` `w_0`
`\vdots`
`d_{T-1}` `r_{T-1}` `c_{T-1}` `h_{T-1}` `w_{T-1}`

```

Show example

### Input Generation

Let `N=20`.

Let `\mathrm{rand}(L,U)` be a function that generates an integer uniformly at random between `L` and `U`, inclusive.

### Wall Generation

A point located at a corner of a cell is called a "vertex."

Generate `W=\mathrm{rand}(0,N-1)`.
Start with no walls between cells and walls only along the outer boundary of the grid, and repeat the following procedure `W` times.
Choose one vertex uniformly at random from among the vertices that are not adjacent to any wall.
Next, choose one of the four directions—up, down, left, or right—uniformly at random.
Extend a wall from the chosen vertex in the chosen direction until it reaches the outer boundary of the grid or an existing wall.

### Initial Grid Generation

The initial card arrangement is generated uniformly at random from all permutations of `0,1,\dots,N^2-1`.
If `a_{i,j}=iN+j` for every `(i,j)`, regenerate the arrangement.

### Tools (Input generator and visualizer)

- Web version: This is more powerful than the local version providing animations.
Local version: You need a compilation environment of Rust language.
- Pre-compiled binary for Windows: If you are not familiar with the Rust language environment, please use this instead.

Please be aware that sharing visualization results or discussing solutions/ideas during the contest is prohibited.

### Use of Generative AI

Under the current AtCoder Heuristic Contest Generative AI Usage Rules — Version 20250616, it is prohibited to have an AI agent execute a solution program and automatically repeat improvements based on the execution results.

However, with the latest generative AI systems, the distinction between conversational services such as ChatGPT and AI agents such as Codex has become increasingly blurred. Even conversational services may, without an explicit instruction from the user, internally generate test cases, execute a solution program, and automatically repeat improvements based on the execution results.

Some participants already explicitly instruct generative AI systems not to engage in such behavior. To ensure fairness among participants and to enforce the intent of the current rules, anyone using generative AI in this contest must provide the instruction below either at the beginning of each chat or through a custom instruction, project instruction, or instruction file automatically loaded by the relevant generative AI tool, such as AGENTS.md or CLAUDE.md.

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
