# A - Colorful Ouroboros

### Story

Takahashi has a pet snake named "Ouroboros".
When fed, this snake grows its tail by 1 meter, and the newly grown part takes on the color of the food.
It can also bite off its own body, and the tail part that was bitten off becomes food.
Takahashi planned to feed Ouroboros in sequence so that the order of its body colors would match his preference.
However, he accidentally dropped the food, scattering it all over the room.
Your task is to give instructions to Ouroboros so that it eats all the food and make the final order of body colors as close as possible to Takahashi's preference.
Fewer instructions are better.

### Problem Statement

There is an `N \times N` grid.
The coordinate of the top-left cell is `(0,0)`, and the coordinate of the cell obtained by moving `i` cells downward and `j` cells to the right from there is `(i,j)`.
There are `M-5` pieces of food placed on the grid, and each piece has a color represented by an integer from `1` to `C`.

A snake moves on this grid.
A snake of length `k` is represented by a sequence of `k` coordinates `(i_0,j_0),\cdots,(i_{k-1},j_{k-1})` and colors `c_0,\cdots,c_{k-1}`, from head to tail.
`(i_0,j_0)` is called the head, `(i_1,j_1),\cdots,(i_{k-2},j_{k-2})` the body, and `(i_{k-1},j_{k-1})` the tail.
The tail may share the same coordinate with another part, but all other coordinates must be distinct.
Initially, the snake has length `5`, all colors are `1`, and it occupies `(4,0),(3,0),(2,0),(1,0),(0,0)`.
No food is placed on the cells occupied by the snake in the initial state.
The following operation is repeated for at most `100000` turns.

In each turn, you specify one of the four directions, up, down, left, or right, and then the snake acts in the following order.

Move:
   First, the snake moves one cell in that direction.
   Let `(i_0',j_0')` be the coordinate reached by moving one cell from the head in direction `d`. Then, after the move, the coordinates of the snake become `(i_0',j_0'),(i_0,j_0),\cdots,(i_{k-2},j_{k-2})`.
   A direction that results in a U-turn, that is, a direction such that `(i_0',j_0')=(i_1,j_1)`, or a direction that goes outside the `N \times N` grid, cannot be specified.
Eating:
   If there is a piece of food of color `c'` at the destination cell, the length of the snake increases by `1`, its coordinates become `(i_0',j_0'),(i_0,j_0),\cdots,(i_{k-2},j_{k-2}),(i_{k-1},j_{k-1})`, and its colors become `c_0,\cdots,c_{k-1},c'`.
Biting off:
   Consider the case where, after moving, the head is on a cell occupied by the body.
   Let its index be `h`. That is, suppose `1\leq h\leq k-2` and `(i_0',j_0')=(i_h',j_h')` holds for the coordinate sequence after the move, `(i_0',j_0'),\cdots,(i_{k-1}',j_{k-1}')`.
   In this case, the snake bites off its own body, its length becomes `h+1`, and the remaining tail side turns into food with the corresponding colors.
   More precisely, the coordinates of the snake become `(i_0',j_0'),\cdots,(i_h',j_h')`, and for each `p=h+1,\cdots,k-1`, a piece of food of color `c_p` is placed at `(i_p',j_p')`.
   After biting off, the tail and the head share the same coordinate.
   Note that eating and biting off never occur in the same turn.

A preferred color sequence `d_0,\cdots,d_{M-1}` is given.
Your task is to make the final color sequence of the snake as close as possible to the preferred color sequence, using as short a sequence of operations as possible.

### Scoring

Let `T` be the length of the output operation sequence, `k` be the length of the snake at the end of the operations, and `c_0,\cdots,c_{k-1}` be the color sequence of the snake.
Here, `k` may be less than `M`.
Also, let the error `E` be the number of indices `p=0,\cdots,k-1` such that `c_p\neq d_p`.
Then, the following absolute score is obtained.

T + 10000 \times (E + 2(M-k))

A smaller absolute score is better.

For each test case, you will obtain a rank score according to your rank determined by lower absolute score. The score of the submission is the total rank score for each test case. The rank score is calculated as follows, and the higher the rank score, the better.

Let `n_{submit}` be the number of contestants with submissions, `n_{lose}` be the number of contestants who received an absolute score lower than yours, and `n_{tie}` be the number of other contestants who received an absolute score equal to yours. Then your rank in this test case is determined as `r=n_{lose}+0.5 n_{tie}`, and your rank score is `\mathrm{round}(10^9\times (1-\frac{r}{n_{submit}}))`.

The final ranking will be determined by the system test with more inputs which will be run after the contest is over.
In both the provisional/system test, if your submission produces illegal output or exceeds the time limit for some test cases, only the rank score for those test cases will be zero.
The system test will be performed only for the last submission which received a result other than CE .
Be careful not to make a mistake in the final submission.
Number of test cases

- Provisional test: 50
- System test: 2000. We will publish seeds.txt (sha256=daf8cb67f5deb3bcb35e3d34b97a3b7b7a3ee8da1fe1cbe24f39389ad64a6c85) after the contest is over.

About relative evaluation system
In both the provisional/system test, the standings will be calculated using only the last submission which received a result other than CE.

The scores shown in the standings are relative, and whenever a new submission arrives, all relative scores are recalculated.
On the other hand, the score for each submission shown on the submissions page is the sum of the absolute score for each test case, and the relative scores are not shown.
In order to know the relative score of submission other than the latest one in the current standings, you need to resubmit it.
If your submission produces illegal output or exceeds the time limit for some test cases, the score shown on the submissions page will be 0, but the standings show the sum of the relative scores for the test cases that were answered correctly.
About execution time
Execution time may vary slightly from run to run.
In addition, since system tests simultaneously perform a large number of executions, it has been observed that execution time increases by several percent compared to provisional tests.
For these reasons, submissions that are very close to the time limit may result in TLE in the system test.
Please measure the execution time in your program to terminate the process, or have enough margin in the execution time.

---

### Input

The input is given from Standard Input in the following format.

```
`N` `M` `C`
`d_0` `\cdots` `d_{M-1}`
`f_{0,0}` `\cdots` `f_{0,N-1}`
`\vdots`
`f_{N-1,0}` `\cdots` `f_{N-1,N-1}`

```

- `N`, the size of the grid, is an integer between `8` and `16`, inclusive.
- `M` represents the length of the preferred color sequence, and is an integer between `N^2/4` and `3N^2/4`, inclusive.
- `C`, the number of colors, is an integer between `3` and `7`, inclusive.
- `d_0,\cdots,d_{M-1}` represents the preferred color sequence, and each `d_p` is an integer between `1` and `C`, inclusive. It is guaranteed that `d_0=\cdots=d_4=1`.
`f_{i,j}` is an integer between `0` and `C` representing the color of the food placed at cell `(i,j)` in the initial state.
- If `f_{i,j}=0`, no food is placed on that cell.
- If `1 \leq f_{i,j} \leq C`, a piece of food of color `f_{i,j}` is placed on that cell.
- It is guaranteed that `f_{0,0}=\cdots=f_{4,0}=0`.
- The number of cells with food is exactly `M-5`.

### Output

Let `T` be the number of operations.
Let `a_t` denote the direction of movement on turn `t`, represented by one of the characters U, D, L, and R, corresponding to up, down, left, and right.
Then, output to Standard Output in the following format.

```
`a_0`
`\vdots`
`a_{T-1}`

```

Show example

### Sample Solution (Python)

A sample solution in Python is shown below.
This program eats all the food in order while moving in a vertical zigzag pattern.

```
import sysinput = sys.stdin.readline N, M, C = map(int, input().split())d = list(map(int, input().split()))f = [list(map(int, input().split())) for _ in range(N)] ans = [] for _ in range(4, N - 1):    ans.append('D') for col in range(1, N):    ans.append('R')    if col % 2 == 1:        for _ in range(N - 1):            ans.append('U')    else:        for _ in range(N - 1):            ans.append('D') print('\n'.join(ans))
```

```
import sys
input = sys.stdin.readline

N, M, C = map(int, input().split())
d = list(map(int, input().split()))
f = [list(map(int, input().split())) for _ in range(N)]

ans = []

for _ in range(4, N - 1):
    ans.append('D')

for col in range(1, N):
    ans.append('R')
    if col % 2 == 1:
        for _ in range(N - 1):
            ans.append('U')
    else:
        for _ in range(N - 1):
            ans.append('D')

print('\n'.join(ans))

```

### Input Generation

Let `\mathrm{rand}(L, U)` be a function that generates an integer uniformly at random between `L` and `U`, inclusive.

Generation of `N`, `M`, and `C`

- `N=\mathrm{rand}(8,16)`
- `M=\mathrm{rand}(\lceil\frac{N^2}{4}\rceil,\lfloor\frac{3N^2}{4}\rfloor)`
- `C=\mathrm{rand}(3,7)`

Generation of `d`
Set `d_0=\cdots=d_4=1`, and generate the remaining `M-5` values as follows.

Let `n_0=0` and `n_C=M-5-C`.
Generate `C-1` random numbers `n_1,\cdots,n_{C-1}` by `\mathrm{rand}(0,M-5-C)`, and sort them in ascending order.
Using these, define the number of occurrences of each color `m_c` by `m_c=n_c-n_{c-1}+1`.
If there exists a `c` such that `m_c>(M-5)/2`, generate `n` again from scratch.
Prepare `m_c` copies of each color `c`, and shuffle them in random order to obtain `d_5,\cdots,d_{M-1}`.

Generation of `f`
Choose `M-5` cells uniformly at random from the `N^2-5` cells other than the five cells occupied by the snake in the initial state, `(4,0),(3,0),(2,0),(1,0),(0,0)`, and denote them by `(i_5,j_5),\cdots,(i_{M-1},j_{M-1})`.
Let `d_5',\cdots,d_{M-1}'` be a random permutation of `d_5,\cdots,d_{M-1}`.
Using these, set `f_{i_p,j_p}=d_p'` for each `p=5,\cdots,M-1`.

### Tools (Input generator and visualizer)

- Web version: This is more powerful than the local version providing animations and manual play.
Local version: You need a compilation environment of Rust language.
- Pre-compiled binary for Windows: If you are not familiar with the Rust language environment, please use this instead.

Please be aware that sharing visualization results or discussing solutions/ideas during the contest is prohibited.

