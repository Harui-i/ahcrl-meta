use tools::{gen, parse_input, Input, State};

use crate::official_compat::{decide_ai_move, legal_mask, official_score, update_state};
use crate::particle_filter::ParticleFilterSmc;

pub const DEFAULT_PF_PARTICLES: usize = 16;

pub struct EnvSlot {
    pub input: Input,
    pub state: State,
    pub pfilters: Vec<ParticleFilterSmc>,
    pub turn: usize,
    pub done: bool,
    pub prev_score: i64,
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

    pub fn from_input_text(text: &str) -> Self {
        Self::new(parse_input(text))
    }

    pub fn new(input: Input) -> Self {
        Self::new_with_seed(input, DEFAULT_PF_PARTICLES, 0)
    }

    pub fn new_with_seed(input: Input, pf_particles: usize, seed: u64) -> Self {
        let state = State::new(&input);
        let prev_score = official_score(&input, &state);
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
        let mask = legal_mask(&self.input, &self.state, 0);
        if action >= mask.len() || !mask[action] {
            return Err(format!("invalid player0 action index {}", action));
        }

        let mut moves = vec![action_xy];
        for i in 1..self.input.M {
            moves.push(decide_ai_move(&self.input, &self.state, i - 1, self.turn));
        }
        for i in 1..self.input.M {
            self.pfilters[i - 1].update(&self.input, &self.state, i, moves[i]);
        }
        self.state = update_state(&self.input, &self.state, &moves)?;
        self.action_history.push(action_xy);
        self.turn += 1;

        let score = official_score(&self.input, &self.state);
        self.reward = (score - self.prev_score) as f64 / 100000.0;
        self.prev_score = score;
        self.done = self.turn >= self.input.T;
        Ok(())
    }

    pub fn score(&self) -> i64 {
        official_score(&self.input, &self.state)
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
