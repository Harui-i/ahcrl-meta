use ahcrl_env_core::{
    write_f16_slice, write_f32_slice, ContestEnv, DType, EnvFactory, EnvSpec, StepOutcome,
    TensorSpec, VisualizerData, PROTOCOL_VERSION,
};
use serde::Deserialize;
use serde_json::Value;
use tools::rl_bridge::{state_view, StateView};
use tools::{gen, Input, State, DIR};

pub const MAX_BOARD_SIZE: usize = 16;
pub const MAX_COLORS: usize = 7;
pub const ACTION_COUNT: usize = 4;
pub const MAX_SEQUENCE_LENGTH: usize = 192;
pub const BOARD_FEATURE_COUNT: usize = 8;
pub const GLOBAL_FEATURE_COUNT: usize = 10;
pub const ACTION_FEATURE_COUNT: usize = 7;
pub const POSITION_SENTINEL: u8 = 255;
pub const INITIAL_SNAKE_LENGTH: usize = 5;
const MAX_OFFICIAL_STEPS: usize = 100_000;
const DEFAULT_MAX_STEPS_PER_CELL: usize = 4;

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ahc063Config {
    pub fixed_n: Option<usize>,
    pub fixed_m: Option<usize>,
    pub fixed_c: Option<usize>,
    pub max_steps_per_cell: usize,
}

impl Default for Ahc063Config {
    fn default() -> Self {
        Self {
            fixed_n: None,
            fixed_m: None,
            fixed_c: None,
            max_steps_per_cell: DEFAULT_MAX_STEPS_PER_CELL,
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
        if self.max_steps_per_cell == 0 {
            return Err(format!(
                "max_steps_per_cell must be positive, got {}",
                self.max_steps_per_cell
            ));
        }
        Ok(())
    }
}

pub struct Ahc063Factory {
    config: Ahc063Config,
}

impl EnvFactory for Ahc063Factory {
    type Env = Ahc063Env;

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
                    name: "board_food".to_owned(),
                    dtype: DType::U8,
                    shape: vec![MAX_BOARD_SIZE, MAX_BOARD_SIZE],
                },
                TensorSpec {
                    name: "board_features".to_owned(),
                    dtype: DType::F16,
                    shape: vec![BOARD_FEATURE_COUNT, MAX_BOARD_SIZE, MAX_BOARD_SIZE],
                },
                TensorSpec {
                    name: "slot_colors".to_owned(),
                    dtype: DType::U8,
                    shape: vec![MAX_SEQUENCE_LENGTH, 2],
                },
                TensorSpec {
                    name: "slot_positions".to_owned(),
                    dtype: DType::U8,
                    shape: vec![MAX_SEQUENCE_LENGTH, 2],
                },
                TensorSpec {
                    name: "global_features".to_owned(),
                    dtype: DType::F16,
                    shape: vec![GLOBAL_FEATURE_COUNT],
                },
                TensorSpec {
                    name: "previous_action".to_owned(),
                    dtype: DType::U8,
                    shape: vec![1],
                },
                TensorSpec {
                    name: "action_colors".to_owned(),
                    dtype: DType::U8,
                    shape: vec![ACTION_COUNT],
                },
                TensorSpec {
                    name: "action_features".to_owned(),
                    dtype: DType::F16,
                    shape: vec![ACTION_COUNT, ACTION_FEATURE_COUNT],
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
        Ahc063Env::new(input, self.config.max_steps_per_cell)
    }
}

pub struct Ahc063Env {
    pub input: Input,
    pub state: State,
    pub actions: Vec<usize>,
    max_steps: usize,
    previous_action: Option<usize>,
    best_score: i64,
    done: bool,
}

impl Ahc063Env {
    pub fn from_seed(seed: u64, config: &Ahc063Config) -> Result<Self, String> {
        config.validate()?;
        let factory = Ahc063Factory {
            config: config.clone(),
        };
        factory.create(seed)
    }

    pub fn new(input: Input, max_steps_per_cell: usize) -> Result<Self, String> {
        let state = State::new(&input);
        let max_steps = max_steps_per_cell
            .checked_mul(input.N)
            .and_then(|value| value.checked_mul(input.N))
            .ok_or_else(|| "max_steps_per_cell * N^2 overflowed usize".to_owned())?;
        if max_steps > MAX_OFFICIAL_STEPS {
            return Err(format!(
                "max_steps_per_cell * N^2 must be at most {MAX_OFFICIAL_STEPS}, got {max_steps}"
            ));
        }
        let best_score = state.score();
        Ok(Self {
            input,
            state,
            actions: Vec::new(),
            max_steps,
            previous_action: None,
            best_score,
            done: false,
        })
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

    fn board_food(&self) -> Vec<u8> {
        let view = self.state_view();
        let mut output = vec![0_u8; MAX_BOARD_SIZE * MAX_BOARD_SIZE];
        for row in 0..self.input.N {
            for col in 0..self.input.N {
                output[row * MAX_BOARD_SIZE + col] = view.food[row][col] as u8;
            }
        }
        output
    }

    fn state_view(&self) -> StateView<'_> {
        state_view(&self.state)
    }

    fn board_features(&self) -> Vec<f32> {
        let view = self.state_view();
        let mut features = vec![0.0_f32; BOARD_FEATURE_COUNT * MAX_BOARD_SIZE * MAX_BOARD_SIZE];
        let index = |channel: usize, row: usize, col: usize| {
            (channel * MAX_BOARD_SIZE + row) * MAX_BOARD_SIZE + col
        };
        let (head_row, head_col) = view.positions[0];
        for row in 0..self.input.N {
            for col in 0..self.input.N {
                features[index(0, row, col)] = 1.0;
                features[index(5, row, col)] = (row as f32 - head_row as f32) / 15.0;
                features[index(6, row, col)] = (col as f32 - head_col as f32) / 15.0;
                features[index(7, row, col)] = ((row as f32 - head_row as f32).abs()
                    + (col as f32 - head_col as f32).abs())
                    / 30.0;
            }
        }
        for (segment, (&(row, col), _color)) in view.positions.iter().zip(view.colors).enumerate() {
            features[index(1, row, col)] = 1.0;
            features[index(2, row, col)] = f32::from(segment == 0);
            features[index(3, row, col)] =
                f32::from(segment > 0 && segment + 1 < view.positions.len());
            features[index(4, row, col)] = f32::from(segment + 1 == view.positions.len());
        }
        features
    }

    fn slot_colors(&self) -> Vec<u8> {
        let view = self.state_view();
        let mut output = vec![0_u8; MAX_SEQUENCE_LENGTH * 2];
        for p in 0..MAX_SEQUENCE_LENGTH {
            if p < self.input.M {
                output[2 * p] = self.input.d[p] as u8;
            }
            if p < view.colors.len() {
                output[2 * p + 1] = view.colors[p] as u8;
            }
        }
        output
    }

    fn slot_positions(&self) -> Vec<u8> {
        let view = self.state_view();
        let mut output = vec![POSITION_SENTINEL; MAX_SEQUENCE_LENGTH * 2];
        for (p, &(row, col)) in view.positions.iter().take(MAX_SEQUENCE_LENGTH).enumerate() {
            output[2 * p] = row as u8;
            output[2 * p + 1] = col as u8;
        }
        output
    }

    fn error_count(&self, view: &StateView<'_>) -> usize {
        view.colors
            .iter()
            .zip(&self.input.d)
            .filter(|(actual, desired)| actual != desired)
            .count()
    }

    fn global_features(&self) -> Vec<f32> {
        let view = self.state_view();
        let food_count = view
            .food
            .iter()
            .flatten()
            .filter(|&&value| value != 0)
            .count();
        let errors = self.error_count(&view);
        let correct_prefix = view
            .colors
            .iter()
            .zip(&self.input.d)
            .take_while(|(a, d)| a == d)
            .count();
        let denominator = (20_000 * self.input.M + self.max_steps).max(1) as f32;
        vec![
            self.input.N as f32 / 16.0,
            self.input.M as f32 / 192.0,
            self.input.C as f32 / 7.0,
            view.positions.len() as f32 / self.input.M.max(1) as f32,
            food_count as f32 / (self.input.M.saturating_sub(INITIAL_SNAKE_LENGTH)).max(1) as f32,
            view.turn as f32 / self.max_steps.max(1) as f32,
            (self.max_steps.saturating_sub(view.turn)) as f32 / self.max_steps.max(1) as f32,
            errors as f32 / self.input.M.max(1) as f32,
            correct_prefix as f32 / self.input.M.max(1) as f32,
            (self.state.score() - self.best_score) as f32 / denominator,
        ]
    }

    fn previous_action_tensor(&self) -> Vec<u8> {
        vec![self.previous_action.map_or(0, |action| action + 1) as u8]
    }

    fn preview(&self, action: usize) -> Option<ActionPreview> {
        if self.legal_mask()[action] == 0 {
            return None;
        }
        let view = self.state_view();
        let (row, col) = view.positions[0];
        let (dr, dc) = [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)][action];
        let destination = ((row as i32 + dr) as usize, (col as i32 + dc) as usize);
        let old_length = view.positions.len();
        let food = view.food[destination.0][destination.1];
        let mut length = if food != 0 {
            old_length + 1
        } else {
            old_length
        };
        let mut collision_index = 0;
        if food == 0 {
            for h in 1..=old_length.saturating_sub(2) {
                if view.positions[h - 1] == destination {
                    collision_index = h;
                    length = h + 1;
                    break;
                }
            }
        }
        let mut new_errors: i32 = 0;
        for p in 0..length {
            let color = if p < old_length { view.colors[p] } else { food };
            if self.input.d[p] != color {
                new_errors += 1;
            }
        }
        let old_errors = self.error_count(&view) as i32;
        let score_after = (view.turn + 1) as i64
            + 10_000 * (new_errors as i64 + 2 * (self.input.M as i64 - length as i64));
        Some(ActionPreview {
            food,
            collision_index,
            length,
            food_match: food != 0 && old_length < self.input.M && food == self.input.d[old_length],
            body_collision: collision_index != 0,
            error_delta: new_errors - old_errors,
            score_delta: score_after - self.state.score(),
        })
    }

    fn action_colors(&self) -> Vec<u8> {
        (0..ACTION_COUNT)
            .map(|action| self.preview(action).map_or(0, |p| p.food as u8))
            .collect()
    }

    fn action_features(&self) -> Vec<f32> {
        let mut output = vec![0.0_f32; ACTION_COUNT * ACTION_FEATURE_COUNT];
        for action in 0..ACTION_COUNT {
            if let Some(preview) = self.preview(action) {
                let offset = action * ACTION_FEATURE_COUNT;
                output[offset] = f32::from(preview.food_match);
                output[offset + 1] = f32::from(preview.body_collision);
                output[offset + 2] = preview.collision_index as f32 / self.input.M.max(1) as f32;
                output[offset + 3] = preview.length as f32 / self.input.M.max(1) as f32;
                output[offset + 4] =
                    (preview.length as i32 - self.state_view().positions.len() as i32) as f32
                        / self.input.M.max(1) as f32;
                output[offset + 5] = preview.error_delta as f32 / self.input.M.max(1) as f32;
                output[offset + 6] =
                    preview.score_delta as f32 / (20_000 * self.input.M + 1) as f32;
            }
        }
        output
    }
}

#[derive(Clone, Copy)]
struct ActionPreview {
    food: usize,
    collision_index: usize,
    length: usize,
    food_match: bool,
    body_collision: bool,
    error_delta: i32,
    score_delta: i64,
}

impl ContestEnv for Ahc063Env {
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

    fn initial_outcome(&self) -> StepOutcome {
        StepOutcome {
            reward: 0.0,
            done: self.done,
            score: self.best_score,
        }
    }

    fn step(&mut self, action: u32) -> Result<StepOutcome, String> {
        self.validate_action(action)?;
        let action = action as usize;
        self.state.apply(action)?;
        self.actions.push(action);
        self.previous_action = Some(action);
        let current_score = self.state.score();
        let reward = if current_score < self.best_score {
            let reward = (self.best_score - current_score) as f32 / 10_000.0;
            self.best_score = current_score;
            reward
        } else {
            0.0
        };
        let view = state_view(&self.state);
        self.done = is_complete(&self.input, &view) || view.turn >= self.max_steps;
        Ok(StepOutcome {
            reward,
            done: self.done,
            score: self.best_score,
        })
    }

    fn write_observation(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
        match name {
            "board_food" => {
                let values = self.board_food();
                if destination.len() != values.len() {
                    return Err("board_food destination size mismatch".to_owned());
                }
                destination.copy_from_slice(&values);
                Ok(())
            }
            "board_features" => write_f16_slice(&self.board_features(), destination),
            "slot_colors" => {
                let values = self.slot_colors();
                if destination.len() != values.len() {
                    return Err("slot_colors destination size mismatch".to_owned());
                }
                destination.copy_from_slice(&values);
                Ok(())
            }
            "slot_positions" => {
                let values = self.slot_positions();
                if destination.len() != values.len() {
                    return Err("slot_positions destination size mismatch".to_owned());
                }
                destination.copy_from_slice(&values);
                Ok(())
            }
            "global_features" => write_f16_slice(&self.global_features(), destination),
            "previous_action" => {
                let values = self.previous_action_tensor();
                if destination.len() != values.len() {
                    return Err("previous_action destination size mismatch".to_owned());
                }
                destination.copy_from_slice(&values);
                Ok(())
            }
            "action_colors" => {
                let values = self.action_colors();
                if destination.len() != values.len() {
                    return Err("action_colors destination size mismatch".to_owned());
                }
                destination.copy_from_slice(&values);
                Ok(())
            }
            "action_features" => write_f16_slice(&self.action_features(), destination),
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

    fn visualizer_data(&self) -> Result<VisualizerData, String> {
        Ok(VisualizerData {
            input: self.input.to_string(),
            output: self.output_text(),
        })
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
        let slot = Ahc063Env::from_seed(0, &default_config()).unwrap();
        let mask = slot.legal_mask();
        for (action, &legal) in mask.iter().enumerate() {
            let mut candidate = Ahc063Env::from_seed(0, &default_config()).unwrap();
            assert_eq!(candidate.state.apply(action).is_ok(), legal != 0);
            assert_eq!(candidate.validate_action(action as u32).is_ok(), legal != 0);
        }
    }

    #[test]
    fn trajectories_track_official_prefix_best_score_and_reward_delta() {
        let mut saw_positive_reward = false;
        let mut saw_final_score_above_best = false;
        for seed in [0_u64, 1, 3, 99] {
            let mut slot = Ahc063Env::from_seed(seed, &default_config()).unwrap();
            let mut outcome = slot.initial_outcome();
            let initial_score = outcome.score;
            let mut expected_best_score = initial_score;
            let mut reward_sum = 0.0_f32;
            for turn in 0..512 {
                if outcome.done {
                    break;
                }
                let candidates = slot
                    .legal_mask()
                    .iter()
                    .enumerate()
                    .filter_map(|(action, &legal)| (legal != 0).then_some(action))
                    .collect::<Vec<_>>();
                let action = candidates[(turn * 17 + seed as usize) % candidates.len()];
                outcome = slot.step(action as u32).unwrap();
                reward_sum += outcome.reward;
                saw_positive_reward |= outcome.reward > 0.0;
                let (prefix_score, error, _) = compute_score_details(&slot.input, &slot.actions);
                assert_eq!(error, "");
                expected_best_score = expected_best_score.min(prefix_score);
                assert_eq!(outcome.score, expected_best_score);
                assert!(outcome.reward >= 0.0);
            }
            let expected_reward = (initial_score - expected_best_score) as f32 / 10_000.0;
            assert!((reward_sum - expected_reward).abs() < 1e-3);
            assert_eq!(slot.output_text().lines().count(), slot.actions.len());
            let (final_score, error, _) = compute_score_details(&slot.input, &slot.actions);
            assert_eq!(error, "");
            saw_final_score_above_best |= final_score > outcome.score;
        }
        assert!(saw_positive_reward);
        assert!(saw_final_score_above_best);
    }

    #[test]
    fn global_feature_tracks_current_score_above_best() {
        let mut slot = Ahc063Env::from_seed(0, &default_config()).unwrap();
        let view = state_view(&slot.state);
        let (row, col) = view.positions[0];
        let action = [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)]
            .iter()
            .enumerate()
            .find_map(|(action, &(dr, dc))| {
                let new_row = (row as i32 + dr) as usize;
                let new_col = (col as i32 + dc) as usize;
                (slot.legal_mask()[action] != 0 && view.food[new_row][new_col] == 0)
                    .then_some(action)
            })
            .unwrap();

        let outcome = slot.step(action as u32).unwrap();

        assert_eq!(outcome.reward, 0.0);
        let features = slot.global_features();
        assert!(
            (features[9]
                - (slot.state.score() - slot.best_score) as f32
                    / (20_000 * slot.input.M + 1) as f32)
                .abs()
                < 1e-7
        );
    }

    #[test]
    fn max_steps_scales_with_board_area() {
        for (n, expected) in [(8, 256), (16, 1024)] {
            let config = Ahc063Config {
                fixed_n: Some(n),
                ..default_config()
            };
            let slot = Ahc063Env::from_seed(0, &config).unwrap();
            assert_eq!(slot.max_steps, expected);
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
            max_steps_per_cell: 1,
            ..default_config()
        };
        let mut slot = Ahc063Env::from_seed(0, &config).unwrap();
        let expected_steps = slot.input.N * slot.input.N;
        let mut outcome = slot.initial_outcome();
        while !outcome.done {
            let action = slot
                .legal_mask()
                .iter()
                .position(|&legal| legal != 0)
                .unwrap();
            outcome = slot.step(action as u32).unwrap();
        }
        assert_eq!(slot.actions.len(), expected_steps);
        assert!(outcome.done);
        assert!(slot.validate_action(0).is_err());
    }

    #[test]
    fn action_preview_matches_official_apply_for_random_trajectories() {
        for seed in [0_u64, 1, 7, 99] {
            let mut slot = Ahc063Env::from_seed(seed, &default_config()).unwrap();
            for turn in 0..128 {
                if slot.done {
                    break;
                }
                let mask = slot.legal_mask();
                let features = slot.action_features();
                for (action, &legal) in mask.iter().enumerate() {
                    if legal == 0 {
                        continue;
                    }
                    let mut actions = slot.actions.clone();
                    actions.push(action);
                    let (_, error, candidate) = compute_score_details(&slot.input, &actions);
                    assert_eq!(error, "");
                    let candidate_view = state_view(&candidate);
                    let offset = action * ACTION_FEATURE_COUNT;
                    let current_view = state_view(&slot.state);
                    let current_errors = slot.error_count(&current_view) as i32;
                    let candidate_errors = slot.error_count(&candidate_view) as i32;
                    assert!(
                        (features[offset + 3]
                            - candidate_view.positions.len() as f32 / slot.input.M as f32)
                            .abs()
                            < 1e-5,
                        "seed={seed} turn={turn} action={action} M={} got={} expected={} len={} features={:?}",
                        slot.input.M,
                        features[offset + 3],
                        candidate_view.positions.len() as f32 / slot.input.M as f32,
                        candidate_view.positions.len(),
                        &features[offset..offset + ACTION_FEATURE_COUNT]
                    );
                    assert!(
                        (features[offset + 4]
                            - (candidate_view.positions.len() as i32
                                - current_view.positions.len() as i32)
                                as f32
                                / slot.input.M as f32)
                            .abs()
                            < 1e-5
                    );
                    assert!(
                        (features[offset + 5]
                            - (candidate_errors - current_errors) as f32 / slot.input.M as f32)
                            .abs()
                            < 1e-5
                    );
                    assert!(
                        (features[offset + 6]
                            - (candidate.score() - slot.state.score()) as f32
                                / (20_000 * slot.input.M + 1) as f32)
                            .abs()
                            < 1e-5
                    );
                }
                let action = mask
                    .iter()
                    .enumerate()
                    .find_map(|(index, &legal)| (legal != 0).then_some(index))
                    .unwrap_or(turn % ACTION_COUNT);
                slot.step(action as u32).unwrap();
            }
        }
    }
}
