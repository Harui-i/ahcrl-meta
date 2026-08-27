use ahcrl_env_core::{
    write_f32_slice, ContestEnv, DType, EnvFactory, EnvSpec, TensorSpec, PROTOCOL_VERSION,
};
use serde::Deserialize;
use serde_json::Value;
use tools::rl_bridge::{state_view, StateView};
use tools::{gen, Input, State, DIR};

pub const MAX_BOARD_SIZE: usize = 16;
pub const MAX_COLORS: usize = 7;
pub const ACTION_COUNT: usize = 4;
pub const NUM_PLANES: usize = 43;
pub const INITIAL_SNAKE_LENGTH: usize = 5;
const MAX_OFFICIAL_STEPS: usize = 100_000;

fn plane_index(plane: usize, row: usize, col: usize) -> usize {
    (plane * MAX_BOARD_SIZE + row) * MAX_BOARD_SIZE + col
}

fn fill_actual_board(planes: &mut [f32], n: usize, plane: usize, value: f32) {
    for row in 0..n {
        for col in 0..n {
            planes[plane_index(plane, row, col)] = value;
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ahc063Config {
    pub fixed_n: Option<usize>,
    pub fixed_m: Option<usize>,
    pub fixed_c: Option<usize>,
    pub max_steps: usize,
}

impl Default for Ahc063Config {
    fn default() -> Self {
        Self {
            fixed_n: None,
            fixed_m: None,
            fixed_c: None,
            max_steps: MAX_OFFICIAL_STEPS,
        }
    }
}

impl Ahc063Config {
    fn validate(&self) -> Result<(), String> {
        if let Some(n) = self.fixed_n {
            if !(8..=MAX_BOARD_SIZE).contains(&n) {
                return Err(format!("fixed_n must be in 8..={MAX_BOARD_SIZE}, got {n}"));
            }
        }
        if let Some(c) = self.fixed_c {
            if !(3..=MAX_COLORS).contains(&c) {
                return Err(format!("fixed_c must be in 3..={MAX_COLORS}, got {c}"));
            }
        }
        if let Some(m) = self.fixed_m {
            let global_min = (8_usize * 8).div_ceil(4);
            let global_max = 3 * MAX_BOARD_SIZE * MAX_BOARD_SIZE / 4;
            if !(global_min..=global_max).contains(&m) {
                return Err(format!(
                    "fixed_m must be in {global_min}..={global_max}, got {m}"
                ));
            }
        }
        if !(1..=MAX_OFFICIAL_STEPS).contains(&self.max_steps) {
            return Err(format!(
                "max_steps must be in 1..={MAX_OFFICIAL_STEPS}, got {}",
                self.max_steps
            ));
        }
        Ok(())
    }
}

pub struct Ahc063Factory {
    config: Ahc063Config,
}

impl EnvFactory for Ahc063Factory {
    type Env = EnvSlot;

    fn from_config(config: Value) -> Result<Self, String> {
        let config: Ahc063Config =
            serde_json::from_value(config).map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(Self { config })
    }

    fn spec(&self) -> EnvSpec {
        EnvSpec {
            protocol_version: PROTOCOL_VERSION,
            observations: vec![
                TensorSpec {
                    name: "planes".to_owned(),
                    dtype: DType::F32,
                    shape: vec![NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE],
                },
                TensorSpec {
                    name: "mask".to_owned(),
                    dtype: DType::U8,
                    shape: vec![ACTION_COUNT],
                },
            ],
            metrics: vec![TensorSpec {
                name: "prefix_match_ratio".to_owned(),
                dtype: DType::F32,
                shape: vec![],
            }],
        }
    }

    fn create(&self, seed: u64) -> Result<Self::Env, String> {
        if let Some(m) = self.config.fixed_m {
            let probe = gen(seed, self.config.fixed_n, None, self.config.fixed_c);
            let min_m = (probe.N * probe.N).div_ceil(4);
            let max_m = 3 * probe.N * probe.N / 4;
            if !(min_m..=max_m).contains(&m) {
                return Err(format!(
                    "fixed_m {m} is outside {min_m}..={max_m} for generated N={}",
                    probe.N
                ));
            }
        }
        let input = gen(
            seed,
            self.config.fixed_n,
            self.config.fixed_m,
            self.config.fixed_c,
        );
        Ok(EnvSlot::new(input, self.config.max_steps))
    }
}

pub struct EnvSlot {
    pub input: Input,
    pub state: State,
    pub actions: Vec<usize>,
    max_steps: usize,
    previous_action: Option<usize>,
    previous_score: i64,
    reward: f32,
    done: bool,
}

impl EnvSlot {
    pub fn from_seed(seed: u64, config: &Ahc063Config) -> Result<Self, String> {
        config.validate()?;
        let factory = Ahc063Factory {
            config: config.clone(),
        };
        factory.create(seed)
    }

    pub fn new(input: Input, max_steps: usize) -> Self {
        let state = State::new(&input);
        let previous_score = state.score();
        Self {
            input,
            state,
            actions: Vec::new(),
            max_steps,
            previous_action: None,
            previous_score,
            reward: 0.0,
            done: false,
        }
    }

    pub fn legal_mask(&self) -> [u8; ACTION_COUNT] {
        let view = state_view(&self.state);
        let mut mask = [0_u8; ACTION_COUNT];
        if self.done {
            return mask;
        }
        let (row, col) = view.positions[0];
        let neck = view.positions.get(1).copied();
        for (action, &(dr, dc)) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)]
            .iter()
            .enumerate()
        {
            let new_row = row as i32 + dr;
            let new_col = col as i32 + dc;
            let on_board = 0 <= new_row
                && new_row < self.input.N as i32
                && 0 <= new_col
                && new_col < self.input.N as i32;
            let new_position = (new_row as usize, new_col as usize);
            mask[action] = u8::from(on_board && neck != Some(new_position));
        }
        mask
    }

    pub fn output_text(&self) -> String {
        let mut output = String::new();
        for &action in &self.actions {
            output.push(DIR[action]);
            output.push('\n');
        }
        output
    }

    fn prefix_match_ratio(&self) -> f32 {
        let view = state_view(&self.state);
        let prefix = view
            .colors
            .iter()
            .zip(&self.input.d)
            .take_while(|(actual, desired)| actual == desired)
            .count();
        prefix as f32 / self.input.M.max(1) as f32
    }

    fn encode_planes(&self) -> Vec<f32> {
        let mut planes = vec![0.0_f32; NUM_PLANES * MAX_BOARD_SIZE * MAX_BOARD_SIZE];
        let view = state_view(&self.state);
        let n = self.input.N;
        let m = self.input.M;
        let c = self.input.C;
        let length = view.positions.len();

        for row in 0..n {
            for col in 0..n {
                let food = view.food[row][col];
                if food != 0 {
                    planes[plane_index(food - 1, row, col)] = 1.0;
                }
            }
        }
        for (index, (&(row, col), &color)) in view.positions.iter().zip(view.colors).enumerate() {
            planes[plane_index(7 + color - 1, row, col)] = 1.0;
            planes[plane_index(14, row, col)] = f32::from(index == 0);
            planes[plane_index(15, row, col)] = f32::from(0 < index && index < length - 1);
            planes[plane_index(16, row, col)] = f32::from(index == length - 1);
        }
        if length < m {
            fill_actual_board(&mut planes, n, 16 + self.input.d[length], 1.0);
        }
        for color in 1..=c {
            let count = self.input.d[length..m]
                .iter()
                .filter(|&&value| value == color)
                .count();
            fill_actual_board(&mut planes, n, 23 + color, count as f32 / m.max(1) as f32);
        }
        let (head_row, head_col) = view.positions[0];
        let food_count = view
            .food
            .iter()
            .flatten()
            .filter(|&&value| value != 0)
            .count();
        let scalar_values = [
            n as f32 / MAX_BOARD_SIZE as f32,
            c as f32 / MAX_COLORS as f32,
            length as f32 / m.max(1) as f32,
            length.min(m) as f32 / m.max(1) as f32,
            head_row as f32 / (n - 1).max(1) as f32,
            head_col as f32 / (n - 1).max(1) as f32,
            food_count as f32 / (m - INITIAL_SNAKE_LENGTH).max(1) as f32,
            view.turn as f32 / self.max_steps.max(1) as f32,
        ];
        for (offset, value) in scalar_values.into_iter().enumerate() {
            fill_actual_board(&mut planes, n, 31 + offset, value);
        }
        if let Some(action) = self.previous_action {
            fill_actual_board(&mut planes, n, 39 + action, 1.0);
        }
        planes
    }
}

impl ContestEnv for EnvSlot {
    fn validate_action(&self, action: u32) -> Result<(), String> {
        if self.done {
            return Err("cannot step a finished environment".to_owned());
        }
        let action = action as usize;
        if action >= ACTION_COUNT || self.legal_mask()[action] == 0 {
            return Err(format!("invalid action {action}"));
        }
        Ok(())
    }

    fn step(&mut self, action: u32) -> Result<(), String> {
        self.validate_action(action)?;
        let action = action as usize;
        self.state.apply(action)?;
        self.actions.push(action);
        self.previous_action = Some(action);
        let score = self.state.score();
        self.reward = (self.previous_score - score) as f32 / 10_000.0;
        self.previous_score = score;
        let view = state_view(&self.state);
        self.done = is_complete(&self.input, &view) || view.turn >= self.max_steps;
        Ok(())
    }

    fn reward(&self) -> f32 {
        self.reward
    }

    fn done(&self) -> bool {
        self.done
    }

    fn score(&self) -> i64 {
        self.state.score()
    }

    fn write_observation(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
        match name {
            "planes" => write_f32_slice(&self.encode_planes(), destination),
            "mask" => {
                let mask = self.legal_mask();
                if destination.len() != mask.len() {
                    return Err(format!(
                        "mask destination has {} bytes, expected {}",
                        destination.len(),
                        mask.len()
                    ));
                }
                destination.copy_from_slice(&mask);
                Ok(())
            }
            _ => Err(format!("unknown observation {name}")),
        }
    }

    fn write_metric(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
        match name {
            "prefix_match_ratio" => write_f32_slice(&[self.prefix_match_ratio()], destination),
            _ => Err(format!("unknown metric {name}")),
        }
    }
}

pub fn is_complete(input: &Input, view: &StateView<'_>) -> bool {
    let no_food = view.food.iter().flatten().all(|&value| value == 0);
    no_food && view.positions.len() == input.M && view.colors == input.d
}

#[cfg(test)]
mod tests {
    use super::*;
    use tools::{compute_score_details, parse_input};

    fn default_config() -> Ahc063Config {
        Ahc063Config::default()
    }

    #[test]
    fn official_seed_zero_matches_checked_in_input() {
        let expected = include_str!("../../tools/in/0000.txt");
        assert_eq!(gen(0, None, None, None).to_string(), expected);
    }

    #[test]
    fn legal_mask_agrees_with_official_apply() {
        let slot = EnvSlot::from_seed(0, &default_config()).unwrap();
        let mask = slot.legal_mask();
        for (action, &legal) in mask.iter().enumerate() {
            let mut candidate = EnvSlot::from_seed(0, &default_config()).unwrap();
            assert_eq!(candidate.state.apply(action).is_ok(), legal != 0);
            assert_eq!(candidate.validate_action(action as u32).is_ok(), legal != 0);
        }
    }

    #[test]
    fn trajectories_match_official_score_and_reward_delta() {
        for seed in [0_u64, 1, 3, 99] {
            let mut slot = EnvSlot::from_seed(seed, &default_config()).unwrap();
            let initial_score = slot.score();
            let mut reward_sum = 0.0_f32;
            for turn in 0..512 {
                if slot.done() {
                    break;
                }
                let candidates = slot
                    .legal_mask()
                    .iter()
                    .enumerate()
                    .filter_map(|(action, &legal)| (legal != 0).then_some(action))
                    .collect::<Vec<_>>();
                let action = candidates[(turn * 17 + seed as usize) % candidates.len()];
                slot.step(action as u32).unwrap();
                reward_sum += slot.reward();
            }
            let (official_score, error, _) = compute_score_details(&slot.input, &slot.actions);
            assert_eq!(error, "");
            assert_eq!(slot.score(), official_score);
            let expected_reward = (initial_score - official_score) as f32 / 10_000.0;
            assert!((reward_sum - expected_reward).abs() < 1e-3);
        }
    }

    #[test]
    fn completion_requires_no_food_full_length_and_matching_colors() {
        let input = parse_input("8 5 3\n1 1 1 1 1\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n");
        let food = vec![vec![0; 8]; 8];
        let positions = vec![(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)];
        let matching = vec![1; 5];
        let wrong = vec![1, 1, 1, 1, 2];
        let matching_view = StateView {
            food: &food,
            positions: &positions,
            colors: &matching,
            turn: 0,
        };
        let wrong_view = StateView {
            food: &food,
            positions: &positions,
            colors: &wrong,
            turn: 0,
        };
        assert!(is_complete(&input, &matching_view));
        assert!(!is_complete(&input, &wrong_view));
    }

    #[test]
    fn max_steps_finishes_episode() {
        let config = Ahc063Config {
            max_steps: 1,
            ..default_config()
        };
        let mut slot = EnvSlot::from_seed(0, &config).unwrap();
        let action = slot
            .legal_mask()
            .iter()
            .position(|&legal| legal != 0)
            .unwrap();
        slot.step(action as u32).unwrap();
        assert!(slot.done());
        assert!(slot.validate_action(action as u32).is_err());
    }
}
