use tools::{gen, Input, State};

use crate::official_compat::{current_scores, get_candidates, legal_mask};
use crate::particle_filter::ParticleFilterSmc;

pub const DEFAULT_PF_PARTICLES: usize = 16;

pub struct EnvSlot {
    pub input: Input,
    pub state: State,
    pub pfilters: Vec<ParticleFilterSmc>,
    pub turn: usize,
    pub done: bool,
    pub prev_score: i64,
    pub score_sums: Vec<i64>,
    pub reward: f64,
    pub action_history: Vec<(usize, usize)>,
}

impl EnvSlot {
    pub fn from_seed(seed: u64, m_opt: Option<usize>, u_opt: Option<usize>) -> Self {
        Self::from_seed_with_pf(seed, m_opt, u_opt, DEFAULT_PF_PARTICLES)
    }

    pub fn from_seed_with_pf(
        seed: u64,
        m_opt: Option<usize>,
        u_opt: Option<usize>,
        pf_particles: usize,
    ) -> Self {
        Self::new_with_seed(gen(seed, m_opt, u_opt), pf_particles, seed)
    }

    pub fn new_with_seed(input: Input, pf_particles: usize, seed: u64) -> Self {
        let state = State::new(&input);
        let score_sums = current_scores(&input, &state);
        let prev_score = official_score_from_sums(&score_sums);
        let pfilters = (1..input.M)
            .map(|player| {
                ParticleFilterSmc::new(
                    pf_particles,
                    seed ^ ((player as u64) << 32) ^ 0xa0761d6478bd642f,
                )
            })
            .collect();
        Self {
            input,
            state,
            pfilters,
            turn: 0,
            done: false,
            prev_score,
            score_sums,
            reward: 0.0,
            action_history: vec![],
        }
    }

    pub fn step_action_index(&mut self, action: usize) -> Result<(), String> {
        if self.done {
            self.reward = 0.0;
            return Ok(());
        }
        let n = self.input.N;
        let action_xy = (action / n, action % n);
        let candidates = (0..self.input.M)
            .map(|player| get_candidates(&self.input, &self.state, player))
            .collect::<Vec<_>>();
        if action >= n * n || !candidates[0].contains(&action_xy) {
            return Err(format!("invalid player0 action index {}", action));
        }

        let mut moves = vec![action_xy];
        for (i, candidates) in candidates.iter().enumerate().skip(1) {
            moves.push(decide_ai_move_from_candidates(
                &self.input,
                &self.state,
                i - 1,
                self.turn,
                candidates,
            ));
        }
        for (i, candidates) in candidates.iter().enumerate().skip(1) {
            self.pfilters[i - 1].update_with_candidates(
                &self.input,
                &self.state,
                i,
                moves[i],
                candidates,
            );
        }
        let (next_state, next_score_sums) =
            update_state_unchecked(&self.input, &self.state, &self.score_sums, &moves);
        self.action_history.push(action_xy);
        self.turn += 1;

        self.state = next_state;
        self.score_sums = next_score_sums;
        let score = official_score_from_sums(&self.score_sums);
        self.reward = (score - self.prev_score) as f64 / 100000.0;
        self.prev_score = score;
        self.done = self.turn >= self.input.T;
        Ok(())
    }

    pub fn score(&self) -> i64 {
        official_score_from_sums(&self.score_sums)
    }

    pub fn mask(&self) -> Vec<bool> {
        if self.done {
            vec![false; self.input.N * self.input.N]
        } else {
            legal_mask(&self.input, &self.state, 0)
        }
    }

    pub fn output_text(&self) -> String {
        let mut out = String::new();
        for &(x, y) in &self.action_history {
            out.push_str(&format!("{} {}\n", x, y));
        }
        out
    }
}

fn decide_ai_move_from_candidates(
    input: &Input,
    state: &State,
    ai_idx: usize,
    turn: usize,
    candidates: &[(usize, usize)],
) -> (usize, usize) {
    let player_id = ai_idx + 1;

    let mut scores = Vec::with_capacity(candidates.len());
    for &(x, y) in candidates {
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
    let best = scores
        .iter()
        .enumerate()
        .filter_map(|(i, &score)| {
            if score >= max_score - tolerance {
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let idx = ((r2 * best.len() as f64).floor() as usize).min(best.len() - 1);
    candidates[best[idx]]
}

fn official_score_from_sums(scores: &[i64]) -> i64 {
    let player0_score = scores[0];
    let mut max_ai_score = 0_i64;
    for &score in scores.iter().skip(1) {
        max_ai_score = max_ai_score.max(score);
    }
    (1e5 * (1.0 + player0_score as f64 / max_ai_score as f64).log2()).round() as i64
}

fn update_state_unchecked(
    input: &Input,
    state: &State,
    score_sums: &[i64],
    moves: &[(usize, usize)],
) -> (State, Vec<i64>) {
    let mut new_state = state.clone();
    new_state.selected = moves.to_vec();
    let mut next_score_sums = score_sums.to_vec();

    let mut temp_pos = moves.to_vec();
    let mut move_counts = vec![0_usize; input.N * input.N];
    for &(x, y) in moves {
        move_counts[x * input.N + y] += 1;
    }

    let mut collected = vec![false; input.M];
    for i in 0..input.M {
        let target_pos = temp_pos[i];
        if move_counts[target_pos.0 * input.N + target_pos.1] >= 2 {
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
        let old_owner = new_state.owner[x][y];
        if old_owner >= 0 {
            next_score_sums[old_owner as usize] -=
                input.V[x][y] as i64 * new_state.level[x][y] as i64;
        }
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
        let new_owner = new_state.owner[x][y];
        if new_owner >= 0 {
            next_score_sums[new_owner as usize] +=
                input.V[x][y] as i64 * new_state.level[x][y] as i64;
        }
    }

    for i in 0..input.M {
        if collected[i] {
            temp_pos[i] = state.pos[i];
        }
    }
    new_state.pos = temp_pos;
    (new_state, next_score_sums)
}
