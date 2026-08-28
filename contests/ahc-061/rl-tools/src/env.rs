//! AHC061 adapter for the shared vector-environment protocol.

use ahcrl_env_core::{
    write_f32_slice, ContestEnv, DType, EnvFactory, EnvSpec, TensorSpec, PROTOCOL_VERSION,
};
use serde::Deserialize;
use serde_json::Value;

use crate::vec_env::{EnvSlot, DEFAULT_PF_PARTICLES};

const BOARD_SIZE: usize = 10;
const MAX_PLAYERS: usize = 8;
const MAX_LEVEL: usize = 5;
const ORACLE_PARAMS_PER_PLAYER: usize = 5;
const PLAYER_AGG_FEATURES: usize = 4;
const NUM_PLANES: usize = 154;
const PLANE_M: usize = 24;
const PLANE_U: usize = 25;
const PLANE_SCORE_RATIO: usize = 26;
const PLANE_SCORE_DIFF: usize = 27;
const PLANE_LEGAL_MASK: usize = 28;
const PLANE_PLAYER_SCORE_START: usize = 29;
const PLANE_ORACLE_PARAM_START: usize = PLANE_PLAYER_SCORE_START + MAX_PLAYERS;
const PLANE_COMP_START: usize = PLANE_ORACLE_PARAM_START + MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER;
const PLANE_REACH_START: usize = PLANE_COMP_START + MAX_PLAYERS;
const PLANE_NEXT_GREEDY_START: usize = PLANE_REACH_START + MAX_PLAYERS;
const PLANE_DIST_OWNER_START: usize = PLANE_NEXT_GREEDY_START + MAX_PLAYERS;
const PLANE_DIST_COMP_START: usize = PLANE_DIST_OWNER_START + MAX_PLAYERS;
const PLANE_DIST_CENTER: usize = PLANE_DIST_COMP_START + MAX_PLAYERS;
const PLANE_X_NORM: usize = PLANE_DIST_CENTER + 1;
const PLANE_Y_NORM: usize = PLANE_X_NORM + 1;
const PLANE_POS0_X_NORM: usize = PLANE_Y_NORM + 1;
const PLANE_POS0_Y_NORM: usize = PLANE_POS0_X_NORM + 1;
const PLANE_PLAYER_AGG_START: usize = PLANE_POS0_Y_NORM + 1;
const PLAYER_AGG_OWNER_LEVEL_SUM: usize = 0;
const PLAYER_AGG_OWNER_LEVEL_VALUE_SUM: usize = 1;
const PLAYER_AGG_COMP_LEVEL_SUM: usize = 2;
const PLAYER_AGG_COMP_LEVEL_VALUE_SUM: usize = 3;
const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;
const ALL_BOARD_BITS: u128 = (1_u128 << BOARD_CELLS) - 1;
const LEFT_EDGE_BITS: u128 = col_bits(0);
const RIGHT_EDGE_BITS: u128 = col_bits(BOARD_SIZE - 1);

type BoolBoard = [bool; BOARD_CELLS];
type FloatBoard = [f32; BOARD_CELLS];

const fn col_bits(col: usize) -> u128 {
    let mut bits = 0_u128;
    let mut row = 0_usize;
    while row < BOARD_SIZE {
        bits |= 1_u128 << (row * BOARD_SIZE + col);
        row += 1;
    }
    bits
}

fn plane_idx(plane: usize, x: usize, y: usize) -> usize {
    plane * BOARD_SIZE * BOARD_SIZE + x * BOARD_SIZE + y
}

fn set_plane_idx(plane_bytes: &mut [u8], idx: usize, value: f32) {
    let bits = f32_to_f16_bits(value).to_le_bytes();
    let offset = idx * std::mem::size_of::<u16>();
    plane_bytes[offset..offset + 2].copy_from_slice(&bits);
}

fn set_plane(plane_bytes: &mut [u8], plane: usize, x: usize, y: usize, value: f32) {
    set_plane_idx(plane_bytes, plane_idx(plane, x, y), value);
}

fn fill_plane(plane_bytes: &mut [u8], plane: usize, value: f32) {
    let bits = f32_to_f16_bits(value).to_le_bytes();
    let start = plane * BOARD_SIZE * BOARD_SIZE * std::mem::size_of::<u16>();
    let end = start + BOARD_SIZE * BOARD_SIZE * std::mem::size_of::<u16>();
    for chunk in plane_bytes[start..end].chunks_exact_mut(2) {
        chunk.copy_from_slice(&bits);
    }
}

fn neighbor_bits(bits: u128) -> u128 {
    (((bits & !RIGHT_EDGE_BITS) << 1)
        | ((bits & !LEFT_EDGE_BITS) >> 1)
        | (bits << BOARD_SIZE)
        | (bits >> BOARD_SIZE))
        & ALL_BOARD_BITS
}

fn flood_component(start_bit: u128, passable: u128) -> u128 {
    let mut visited = start_bit & passable;
    loop {
        let next = visited | (neighbor_bits(visited) & passable);
        if next == visited {
            return visited;
        }
        visited = next;
    }
}

fn bitboard_to_mask(bits: u128) -> BoolBoard {
    let mut mask = [false; BOARD_CELLS];
    let mut rest = bits;
    while rest != 0 {
        let idx = rest.trailing_zeros() as usize;
        mask[idx] = true;
        rest &= rest - 1;
    }
    mask
}

fn bitboard_to_candidates(bits: u128) -> Vec<(usize, usize)> {
    let mut candidates = Vec::with_capacity(bits.count_ones() as usize);
    let mut rest = bits;
    while rest != 0 {
        let idx = rest.trailing_zeros() as usize;
        candidates.push((idx / BOARD_SIZE, idx % BOARD_SIZE));
        rest &= rest - 1;
    }
    candidates
}

fn connected_component_mask_from_bits(start: (usize, usize), owner_bits: u128) -> BoolBoard {
    let start_bit = 1_u128 << (start.0 * BOARD_SIZE + start.1);
    bitboard_to_mask(flood_component(start_bit, owner_bits))
}

fn reach_mask_and_candidates_from_bits(
    start: (usize, usize),
    owner_bits: u128,
    occupied_other_bits: u128,
) -> (BoolBoard, Vec<(usize, usize)>) {
    let start_bit = 1_u128 << (start.0 * BOARD_SIZE + start.1);
    let own_component = flood_component(start_bit, owner_bits);
    let reachable = if own_component == 0 {
        start_bit
    } else {
        own_component | neighbor_bits(own_component)
    } & !occupied_other_bits
        & ALL_BOARD_BITS;
    (
        bitboard_to_mask(reachable),
        bitboard_to_candidates(reachable),
    )
}

fn dist_to_sources(sources: &BoolBoard) -> FloatBoard {
    const INF: i32 = 1 << 20;
    let mut dist = [INF; BOARD_CELLS];
    let mut has_source = false;
    for idx in 0..BOARD_CELLS {
        if sources[idx] {
            dist[idx] = 0;
            has_source = true;
        }
    }
    if !has_source {
        return [1.0; BOARD_CELLS];
    }

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let idx = x * BOARD_SIZE + y;
            let mut d = dist[idx];
            if x > 0 {
                d = d.min(dist[(x - 1) * BOARD_SIZE + y] + 1);
            }
            if y > 0 {
                d = d.min(dist[x * BOARD_SIZE + y - 1] + 1);
            }
            dist[idx] = d;
        }
    }
    for x in (0..BOARD_SIZE).rev() {
        for y in (0..BOARD_SIZE).rev() {
            let idx = x * BOARD_SIZE + y;
            let mut d = dist[idx];
            if x + 1 < BOARD_SIZE {
                d = d.min(dist[(x + 1) * BOARD_SIZE + y] + 1);
            }
            if y + 1 < BOARD_SIZE {
                d = d.min(dist[x * BOARD_SIZE + y + 1] + 1);
            }
            dist[idx] = d;
        }
    }

    let mut out = [0.0_f32; BOARD_CELLS];
    for idx in 0..BOARD_CELLS {
        let d = dist[idx];
        out[idx] = if d >= INF { 1.0 } else { d as f32 / 18.0 };
    }
    out
}

struct EncodedSlot {
    plane_bytes: Vec<u8>,
    critic_oracle_bytes: Vec<u8>,
    mask: [u8; BOARD_CELLS],
}

fn encode_slot(slot: &EnvSlot) -> EncodedSlot {
    let mut plane_bytes =
        vec![0_u8; NUM_PLANES * BOARD_SIZE * BOARD_SIZE * std::mem::size_of::<u16>()];
    let mut critic_oracle_bytes =
        vec![0_u8; MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER * std::mem::size_of::<u16>()];
    let mut value_sum = 0.0_f32;
    let mut scores = vec![0.0_f32; slot.input.M];
    let mut owner_bits = vec![0_u128; slot.input.M];

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let value = slot.input.V[x][y] as f32;
            value_sum += value;
            let owner = slot.state.owner[x][y];
            if owner >= 0 {
                let player = owner as usize;
                scores[player] += value * slot.state.level[x][y] as f32;
                owner_bits[player] |= 1_u128 << (x * BOARD_SIZE + y);
            }
        }
    }
    let mean_value = (value_sum / (BOARD_SIZE * BOARD_SIZE) as f32).max(1.0);
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            set_plane(
                &mut plane_bytes,
                0,
                x,
                y,
                slot.input.V[x][y] as f32 / mean_value,
            );
        }
    }

    let mut player_map = vec![0_usize; slot.input.M];
    player_map[0] = 0;
    let mut enemy_order: Vec<usize> = (1..slot.input.M).collect();
    enemy_order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    for (rank, player) in enemy_order.into_iter().enumerate() {
        player_map[player] = rank + 1;
    }

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let owner = slot.state.owner[x][y];
            let mapped_owner = if owner < 0 {
                1
            } else {
                player_map[owner as usize] + 2
            };
            set_plane(&mut plane_bytes, mapped_owner, x, y, 1.0);
        }
    }

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let level = slot.state.level[x][y];
            if (1..=MAX_LEVEL).contains(&level) {
                set_plane(&mut plane_bytes, 9 + level, x, y, 1.0);
            }
        }
    }

    for (player, &mapped_player) in player_map.iter().enumerate() {
        let (x, y) = slot.state.pos[player];
        if mapped_player < MAX_PLAYERS && x < BOARD_SIZE && y < BOARD_SIZE {
            set_plane(&mut plane_bytes, 15 + mapped_player, x, y, 1.0);
        }
    }

    fill_plane(&mut plane_bytes, 23, (100.0 - slot.turn as f32) / 100.0);
    fill_plane(
        &mut plane_bytes,
        PLANE_M,
        slot.input.M as f32 / MAX_PLAYERS as f32,
    );
    fill_plane(
        &mut plane_bytes,
        PLANE_U,
        slot.input.U as f32 / MAX_LEVEL as f32,
    );

    let player0_score = scores[0];
    let max_ai_score = scores
        .iter()
        .skip(1)
        .fold(0.0_f32, |acc, &score| acc.max(score));
    let total_capacity = (value_sum * slot.input.U.max(1) as f32).max(1.0);
    fill_plane(
        &mut plane_bytes,
        PLANE_SCORE_RATIO,
        player0_score / max_ai_score.max(1.0),
    );
    fill_plane(
        &mut plane_bytes,
        PLANE_SCORE_DIFF,
        (player0_score - max_ai_score) / total_capacity,
    );

    let occupied_pos_bits =
        slot.state
            .pos
            .iter()
            .enumerate()
            .fold(0_u128, |bits, (player, &(x, y))| {
                if player < slot.input.M {
                    bits | (1_u128 << (x * BOARD_SIZE + y))
                } else {
                    bits
                }
            });
    let comp_masks = (0..slot.input.M)
        .map(|player| {
            connected_component_mask_from_bits(slot.state.pos[player], owner_bits[player])
        })
        .collect::<Vec<_>>();
    let reach = (0..slot.input.M)
        .map(|player| {
            let pos_bit =
                1_u128 << (slot.state.pos[player].0 * BOARD_SIZE + slot.state.pos[player].1);
            let occupied_other_bits = occupied_pos_bits & !pos_bit;
            reach_mask_and_candidates_from_bits(
                slot.state.pos[player],
                owner_bits[player],
                occupied_other_bits,
            )
        })
        .collect::<Vec<_>>();
    let mut next_move_planes = Vec::with_capacity(slot.input.M);
    for (player, reach_entry) in reach.iter().enumerate() {
        if player == 0 {
            let mut plane = [0.0_f32; BOARD_CELLS];
            for (idx, &ok) in reach_entry.0.iter().enumerate() {
                plane[idx] = if ok { 1.0 } else { 0.0 };
            }
            next_move_planes.push(plane);
        } else {
            next_move_planes.push(
                slot.pfilters[player - 1].predictive_distribution_board100_for_candidates(
                    &slot.input,
                    &slot.state,
                    player,
                    &reach_entry.1,
                ),
            );
        }
    }

    let mut mask = [0_u8; BOARD_CELLS];
    for (idx, &ok) in reach[0].0.iter().enumerate() {
        mask[idx] = if ok { 1 } else { 0 };
    }
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            set_plane(
                &mut plane_bytes,
                PLANE_LEGAL_MASK,
                x,
                y,
                mask[x * BOARD_SIZE + y] as f32,
            );
        }
    }

    let posterior_means = slot
        .pfilters
        .iter()
        .map(|pf| pf.posterior_mean())
        .collect::<Vec<_>>();
    for player in 0..slot.input.M {
        let mapped_player = player_map[player];
        if mapped_player < MAX_PLAYERS {
            fill_plane(
                &mut plane_bytes,
                PLANE_PLAYER_SCORE_START + mapped_player,
                scores[player] / total_capacity,
            );
            let param_start = PLANE_ORACLE_PARAM_START + mapped_player * ORACLE_PARAMS_PER_PLAYER;
            for param_idx in 0..ORACLE_PARAMS_PER_PLAYER {
                let posterior_value = if player > 0 {
                    let mean = posterior_means[player - 1];
                    (match param_idx {
                        0 => mean.wa,
                        1 => mean.wb,
                        2 => mean.wc,
                        3 => mean.wd,
                        4 => mean.eps,
                        _ => 0.0,
                    }) as f32
                } else {
                    0.0
                };
                let oracle_value = if player > 0 {
                    match param_idx {
                        0 => slot.input.wa[player - 1],
                        1 => slot.input.wb[player - 1],
                        2 => slot.input.wc[player - 1],
                        3 => slot.input.wd[player - 1],
                        4 => slot.input.eps[player - 1],
                        _ => 0.0,
                    }
                } else {
                    0.0
                };
                fill_plane(&mut plane_bytes, param_start + param_idx, posterior_value);
                let feature_idx = mapped_player * ORACLE_PARAMS_PER_PLAYER + param_idx;
                let feature_offset = feature_idx * std::mem::size_of::<u16>();
                critic_oracle_bytes[feature_offset..feature_offset + 2]
                    .copy_from_slice(&f32_to_f16_bits(oracle_value as f32).to_le_bytes());
            }
        }
    }

    for player in 0..slot.input.M {
        let mapped_player = player_map[player];
        if mapped_player >= MAX_PLAYERS {
            continue;
        }
        for idx in 0..BOARD_SIZE * BOARD_SIZE {
            let x = idx / BOARD_SIZE;
            let y = idx % BOARD_SIZE;
            set_plane(
                &mut plane_bytes,
                PLANE_COMP_START + mapped_player,
                x,
                y,
                comp_masks[player][idx] as u8 as f32,
            );
            set_plane(
                &mut plane_bytes,
                PLANE_REACH_START + mapped_player,
                x,
                y,
                reach[player].0[idx] as u8 as f32,
            );
            set_plane(
                &mut plane_bytes,
                PLANE_NEXT_GREEDY_START + mapped_player,
                x,
                y,
                next_move_planes[player][idx],
            );
        }

        let mut owner_source = [false; BOARD_CELLS];
        for x in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                owner_source[x * BOARD_SIZE + y] = slot.state.owner[x][y] == player as i32;
            }
        }
        let dist_owner = dist_to_sources(&owner_source);
        let dist_comp = dist_to_sources(&comp_masks[player]);
        for idx in 0..BOARD_SIZE * BOARD_SIZE {
            let x = idx / BOARD_SIZE;
            let y = idx % BOARD_SIZE;
            set_plane(
                &mut plane_bytes,
                PLANE_DIST_OWNER_START + mapped_player,
                x,
                y,
                dist_owner[idx],
            );
            set_plane(
                &mut plane_bytes,
                PLANE_DIST_COMP_START + mapped_player,
                x,
                y,
                dist_comp[idx],
            );
        }

        let mut owner_level_sum = 0.0_f32;
        let mut owner_level_value_sum = 0.0_f32;
        let mut comp_level_sum = 0.0_f32;
        let mut comp_level_value_sum = 0.0_f32;
        for (idx, &in_component) in comp_masks[player].iter().enumerate() {
            let x = idx / BOARD_SIZE;
            let y = idx % BOARD_SIZE;
            let level = slot.state.level[x][y] as f32;
            let level_value = level * slot.input.V[x][y] as f32;
            if slot.state.owner[x][y] == player as i32 {
                owner_level_sum += level;
                owner_level_value_sum += level_value;
            }
            if in_component {
                comp_level_sum += level;
                comp_level_value_sum += level_value;
            }
        }

        let level_capacity = (BOARD_SIZE * BOARD_SIZE * slot.input.U.max(1)) as f32;
        let agg_start = PLANE_PLAYER_AGG_START + mapped_player * PLAYER_AGG_FEATURES;
        fill_plane(
            &mut plane_bytes,
            agg_start + PLAYER_AGG_OWNER_LEVEL_SUM,
            owner_level_sum / level_capacity.max(1.0),
        );
        fill_plane(
            &mut plane_bytes,
            agg_start + PLAYER_AGG_OWNER_LEVEL_VALUE_SUM,
            owner_level_value_sum / total_capacity,
        );
        fill_plane(
            &mut plane_bytes,
            agg_start + PLAYER_AGG_COMP_LEVEL_SUM,
            comp_level_sum / level_capacity.max(1.0),
        );
        fill_plane(
            &mut plane_bytes,
            agg_start + PLAYER_AGG_COMP_LEVEL_VALUE_SUM,
            comp_level_value_sum / total_capacity,
        );
    }

    let (pos0_x, pos0_y) = slot.state.pos[0];
    let inv_board_span = 1.0_f32 / (BOARD_SIZE - 1) as f32;
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let dx = (x as f32 - 4.5).abs();
            let dy = (y as f32 - 4.5).abs();
            set_plane(&mut plane_bytes, PLANE_DIST_CENTER, x, y, (dx + dy) / 9.0);
            set_plane(
                &mut plane_bytes,
                PLANE_X_NORM,
                x,
                y,
                x as f32 * inv_board_span,
            );
            set_plane(
                &mut plane_bytes,
                PLANE_Y_NORM,
                x,
                y,
                y as f32 * inv_board_span,
            );
            set_plane(
                &mut plane_bytes,
                PLANE_POS0_X_NORM,
                x,
                y,
                pos0_x as f32 * inv_board_span,
            );
            set_plane(
                &mut plane_bytes,
                PLANE_POS0_Y_NORM,
                x,
                y,
                pos0_y as f32 * inv_board_span,
            );
        }
    }

    EncodedSlot {
        plane_bytes,
        critic_oracle_bytes,
        mask,
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x800000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        if (mantissa & round_bit) != 0 {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x00001000) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ahc061Config {
    pub fixed_m: Option<usize>,
    pub fixed_u: Option<usize>,
    pub pf_particles: usize,
}

impl Default for Ahc061Config {
    fn default() -> Self {
        Self {
            fixed_m: None,
            fixed_u: None,
            pf_particles: DEFAULT_PF_PARTICLES,
        }
    }
}

impl Ahc061Config {
    fn validate(&self) -> Result<(), String> {
        if self.pf_particles == 0 {
            return Err("pf_particles must be positive".to_owned());
        }
        if let Some(m) = self.fixed_m {
            if !(2..=MAX_PLAYERS).contains(&m) {
                return Err(format!("fixed_m must be in 2..={MAX_PLAYERS}, got {m}"));
            }
        }
        if let Some(u) = self.fixed_u {
            if !(1..=MAX_LEVEL).contains(&u) {
                return Err(format!("fixed_u must be in 1..={MAX_LEVEL}, got {u}"));
            }
        }
        Ok(())
    }
}

pub struct Ahc061Factory {
    config: Ahc061Config,
}

impl EnvFactory for Ahc061Factory {
    type Env = EnvSlot;

    fn from_config(config: Value) -> Result<Self, String> {
        let config: Ahc061Config =
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
                    shape: vec![NUM_PLANES, BOARD_SIZE, BOARD_SIZE],
                },
                TensorSpec {
                    name: "mask".to_owned(),
                    dtype: DType::U8,
                    shape: vec![BOARD_CELLS],
                },
                TensorSpec {
                    name: "critic_oracle".to_owned(),
                    dtype: DType::F32,
                    shape: vec![MAX_PLAYERS, ORACLE_PARAMS_PER_PLAYER],
                },
            ],
            metrics: vec![],
        }
    }

    fn create(&self, seed: u64) -> Result<Self::Env, String> {
        Ok(EnvSlot::from_seed_with_pf(
            seed,
            self.config.fixed_m,
            self.config.fixed_u,
            self.config.pf_particles,
        ))
    }
}

impl ContestEnv for EnvSlot {
    fn validate_action(&self, action: u32) -> Result<(), String> {
        let action = action as usize;
        if self.done {
            return Err("cannot step a finished environment".to_owned());
        }
        if action >= BOARD_CELLS || !self.mask()[action] {
            return Err(format!("invalid action {action}"));
        }
        Ok(())
    }

    fn step(&mut self, action: u32) -> Result<(), String> {
        self.validate_action(action)?;
        self.step_action_index(action as usize)
    }

    fn reward(&self) -> f32 {
        self.reward as f32
    }

    fn done(&self) -> bool {
        self.done
    }

    fn score(&self) -> i64 {
        self.score()
    }

    fn write_observation(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
        let encoded = encode_slot(self);
        match name {
            "planes" => write_f32_slice(&decode_f16(&encoded.plane_bytes), destination),
            "mask" => {
                if destination.len() != BOARD_CELLS {
                    return Err(format!(
                        "mask destination has {} bytes, expected {BOARD_CELLS}",
                        destination.len()
                    ));
                }
                destination.copy_from_slice(&encoded.mask);
                Ok(())
            }
            "critic_oracle" => {
                write_f32_slice(&decode_f16(&encoded.critic_oracle_bytes), destination)
            }
            _ => Err(format!("unknown observation {name}")),
        }
    }

    fn write_metric(&self, name: &str, _destination: &mut [u8]) -> Result<(), String> {
        Err(format!("unknown metric {name}"))
    }
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;
    let value = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut exponent = -14_i32;
            let mut mantissa = mantissa;
            while (mantissa & 0x400) == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent as u32 + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(value)
}
