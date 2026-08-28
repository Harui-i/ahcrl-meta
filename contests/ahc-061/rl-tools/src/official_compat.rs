//! Compatibility implementation for AHC061 official tools.
//!
//! The official code under `../tools` is kept as a vendor snapshot.  This file
//! mirrors the small set of private routines needed by the RL environment:
//! candidate enumeration and score aggregation. Keep this
//! file parity-tested against `tools::parse_output`/`tools::compute_score_details`
//! instead of editing the official tools crate.

use std::collections::VecDeque;
use tools::{Input, State};

pub fn current_scores(input: &Input, state: &State) -> Vec<i64> {
    let mut scores = vec![0_i64; input.M];
    for i in 0..input.N {
        for j in 0..input.N {
            let owner = state.owner[i][j];
            if owner >= 0 {
                scores[owner as usize] += input.V[i][j] as i64 * state.level[i][j] as i64;
            }
        }
    }
    scores
}

pub fn legal_mask(input: &Input, state: &State, player: usize) -> Vec<bool> {
    let mut mask = vec![false; input.N * input.N];
    for (x, y) in get_candidates(input, state, player) {
        mask[x * input.N + y] = true;
    }
    mask
}

pub fn get_candidates(input: &Input, state: &State, player: usize) -> Vec<(usize, usize)> {
    let mut reachable = vec![];
    let mut visited = vec![vec![false; input.N]; input.N];
    let mut queue = VecDeque::new();

    let start = state.pos[player];
    queue.push_back(start);
    visited[start.0][start.1] = true;

    while let Some((x, y)) = queue.pop_front() {
        let mut ok = true;
        for i in 0..input.M {
            if i != player && state.pos[i] == (x, y) {
                ok = false;
                break;
            }
        }
        if ok {
            reachable.push((x, y));
        }
        if state.owner[x][y] == player as i32 {
            let dirs = [(0, 1), (1, 0), (0, !0), (!0, 0)];
            for &(dx, dy) in &dirs {
                let nx = x.wrapping_add(dx);
                let ny = y.wrapping_add(dy);
                if nx < input.N && ny < input.N && !visited[nx][ny] {
                    visited[nx][ny] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    reachable
}
