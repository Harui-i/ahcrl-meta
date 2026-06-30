//! Compatibility implementation for AHC061 official tools.
//!
//! The official code under `../tools` is kept as a vendor snapshot.  This file
//! mirrors the small set of private routines needed by the RL environment:
//! candidate enumeration, AI move selection, and state transition.  Keep this
//! file parity-tested against `tools::parse_output`/`tools::compute_score_details`
//! instead of editing the official tools crate.

use std::collections::{HashMap, VecDeque};
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

pub fn official_score(input: &Input, state: &State) -> i64 {
    let scores = current_scores(input, state);
    let player0_score = scores[0];
    let mut max_ai_score = 0_i64;
    for &score in scores.iter().skip(1) {
        max_ai_score = max_ai_score.max(score);
    }
    (1e5 * (1.0 + player0_score as f64 / max_ai_score as f64).log2()).round() as i64
}

pub fn legal_mask(input: &Input, state: &State, player: usize) -> Vec<bool> {
    let mut mask = vec![false; input.N * input.N];
    for (x, y) in get_candidates(input, state, player) {
        mask[x * input.N + y] = true;
    }
    mask
}

pub fn decide_ai_move(input: &Input, state: &State, ai_idx: usize, turn: usize) -> (usize, usize) {
    let player_id = ai_idx + 1;
    let candidates = get_candidates(input, state, player_id);

    let mut scores = Vec::with_capacity(candidates.len());
    for &(x, y) in &candidates {
        let owner = state.owner[x][y];
        let level = state.level[x][y];
        let value = input.V[x][y] as f64;

        let score = if owner == -1 {
            value * input.wa[ai_idx]
        } else if owner == player_id as i32 {
            if level < input.U {
                value * input.wb[ai_idx]
            } else {
                0.0
            }
        } else if level == 1 {
            value * input.wc[ai_idx]
        } else {
            value * input.wd[ai_idx]
        };
        scores.push(score);
    }

    let r1 = input.r[ai_idx][2 * (turn % input.T)];
    let r2 = input.r[ai_idx][2 * (turn % input.T) + 1];
    if r1 < input.eps[ai_idx] {
        let idx = ((r2 * candidates.len() as f64).floor() as usize).min(candidates.len() - 1);
        return candidates[idx];
    }

    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tolerance = 1e-9 * max_score.abs().max(1.0);
    let best: Vec<usize> = (0..candidates.len())
        .filter(|&i| scores[i] >= max_score - tolerance)
        .collect();
    let idx = ((r2 * best.len() as f64).floor() as usize).min(best.len() - 1);
    candidates[best[idx]]
}

pub fn update_state(
    input: &Input,
    state: &State,
    moves: &[(usize, usize)],
) -> Result<State, String> {
    let mut new_state = state.clone();
    new_state.selected = moves.to_vec();

    for i in 0..input.M {
        let target = moves[i];
        if !is_valid_move(input, state, i, target) {
            return Err(format!(
                "Player {} attempted invalid move from ({}, {}) to ({}, {}).",
                i, state.pos[i].0, state.pos[i].1, target.0, target.1
            ));
        }
    }

    let mut temp_pos = moves.to_vec();
    let mut move_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for &mv in moves {
        *move_counts.entry(mv).or_insert(0) += 1;
    }

    let mut collected = vec![false; input.M];
    for i in 0..input.M {
        let target_pos = temp_pos[i];
        if move_counts[&target_pos] >= 2 {
            let owner = new_state.owner[target_pos.0][target_pos.1];
            if i as i32 != owner {
                collected[i] = true;
            }
        }
    }

    for i in 0..input.M {
        if collected[i] {
            continue;
        }

        let (x, y) = temp_pos[i];
        let owner = new_state.owner[x][y];
        if owner == -1 {
            new_state.owner[x][y] = i as i32;
            new_state.level[x][y] = 1;
        } else if owner == i as i32 {
            if new_state.level[x][y] < input.U {
                new_state.level[x][y] += 1;
            }
        } else {
            new_state.level[x][y] -= 1;
            if new_state.level[x][y] == 0 {
                new_state.owner[x][y] = i as i32;
                new_state.level[x][y] = 1;
            } else {
                collected[i] = true;
            }
        }
    }

    for i in 0..input.M {
        if collected[i] {
            temp_pos[i] = state.pos[i];
        }
    }
    new_state.pos = temp_pos;
    Ok(new_state)
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

pub fn is_valid_move(input: &Input, state: &State, player: usize, target: (usize, usize)) -> bool {
    if target.0 >= input.N || target.1 >= input.N {
        return false;
    }
    for i in 0..input.M {
        if i != player && state.pos[i] == target {
            return false;
        }
    }
    get_candidates(input, state, player)
        .iter()
        .any(|&candidate| candidate == target)
}
