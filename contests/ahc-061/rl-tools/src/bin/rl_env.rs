use std::io::{self, BufRead, Read, Write};
use std::time::Instant;

use ahc061_rl_tools::vec_env::{EnvSlot, DEFAULT_PF_PARTICLES};
use rayon::prelude::*;

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

fn official_score_from_scores(scores: &[i64]) -> i64 {
    let player0_score = scores[0];
    let mut max_ai_score = 0_i64;
    for &score in scores.iter().skip(1) {
        max_ai_score = max_ai_score.max(score);
    }
    (1e5 * (1.0 + player0_score as f64 / max_ai_score as f64).log2()).round() as i64
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
    critic_posterior_bytes: Vec<u8>,
    critic_oracle_bytes: Vec<u8>,
    mask: [u8; BOARD_CELLS],
    reward: f32,
    done: u8,
    score: i64,
}

fn encode_slot(slot: &EnvSlot) -> EncodedSlot {
    let mut plane_bytes =
        vec![0_u8; NUM_PLANES * BOARD_SIZE * BOARD_SIZE * std::mem::size_of::<u16>()];
    let mut critic_posterior_bytes =
        vec![0_u8; MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER * std::mem::size_of::<u16>()];
    let mut critic_oracle_bytes =
        vec![0_u8; MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER * std::mem::size_of::<u16>()];
    let mut value_sum = 0.0_f32;
    let mut scores = vec![0.0_f32; slot.input.M];
    let mut score_ints = vec![0_i64; slot.input.M];
    let mut owner_bits = vec![0_u128; slot.input.M];

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let value = slot.input.V[x][y] as f32;
            value_sum += value;
            let owner = slot.state.owner[x][y];
            if owner >= 0 {
                let player = owner as usize;
                scores[player] += value * slot.state.level[x][y] as f32;
                score_ints[player] += slot.input.V[x][y] as i64 * slot.state.level[x][y] as i64;
                owner_bits[player] |= 1_u128 << (x * BOARD_SIZE + y);
            }
        }
    }
    let score = official_score_from_scores(&score_ints);
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

    for player in 0..slot.input.M {
        let mapped_player = player_map[player];
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
    for player in 0..slot.input.M {
        if player == 0 {
            let mut plane = [0.0_f32; BOARD_CELLS];
            for (idx, &ok) in reach[player].0.iter().enumerate() {
                plane[idx] = if ok { 1.0 } else { 0.0 };
            }
            next_move_planes.push(plane);
        } else {
            next_move_planes.push(
                slot.pfilters[player - 1].predictive_distribution_board100_for_candidates(
                    &slot.input,
                    &slot.state,
                    player,
                    &reach[player].1,
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
                critic_posterior_bytes[feature_offset..feature_offset + 2]
                    .copy_from_slice(&f32_to_f16_bits(posterior_value).to_le_bytes());
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
        for idx in 0..BOARD_SIZE * BOARD_SIZE {
            let x = idx / BOARD_SIZE;
            let y = idx % BOARD_SIZE;
            let level = slot.state.level[x][y] as f32;
            let level_value = level * slot.input.V[x][y] as f32;
            if slot.state.owner[x][y] == player as i32 {
                owner_level_sum += level;
                owner_level_value_sum += level_value;
            }
            if comp_masks[player][idx] {
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
        critic_posterior_bytes,
        critic_oracle_bytes,
        mask,
        reward: slot.reward as f32,
        done: if slot.done { 1 } else { 0 },
        score,
    }
}

fn encode_slots(slots: &[EnvSlot]) -> Vec<EncodedSlot> {
    slots.par_iter().map(encode_slot).collect()
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

fn write_encoded_obs(slots: &[EnvSlot], out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "OKF16 {} {} {} {} {}",
        slots.len(),
        NUM_PLANES,
        BOARD_SIZE,
        BOARD_SIZE,
        MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER
    )?;
    let encoded = encode_slots(slots);
    for item in &encoded {
        out.write_all(&item.plane_bytes)?;
    }
    for item in &encoded {
        out.write_all(&item.critic_posterior_bytes)?;
    }
    for item in &encoded {
        out.write_all(&item.critic_oracle_bytes)?;
    }

    let mut mask_bytes = Vec::with_capacity(encoded.len() * BOARD_SIZE * BOARD_SIZE);
    for item in &encoded {
        mask_bytes.extend_from_slice(&item.mask);
    }
    out.write_all(&mask_bytes)?;
    for item in &encoded {
        out.write_all(&item.reward.to_le_bytes())?;
    }
    for item in &encoded {
        out.write_all(&[item.done])?;
    }
    for item in &encoded {
        out.write_all(&item.score.to_le_bytes())?;
    }
    writeln!(out, "END")?;
    out.flush()
}

fn first_legal_action(slot: &EnvSlot) -> Result<usize, String> {
    if slot.done {
        return Ok(0);
    }
    slot.mask()
        .iter()
        .position(|&ok| ok)
        .ok_or_else(|| "no legal action".to_string())
}

fn step_first_legal_noobs(slots: &mut [EnvSlot], out: &mut impl Write) -> Result<(), String> {
    slots.par_iter_mut().try_for_each(|slot| {
        let action = first_legal_action(slot)?;
        slot.step_action_index(action)
    })?;
    writeln!(out, "OK_NOOBS").map_err(|err| err.to_string())?;
    out.flush().map_err(|err| err.to_string())
}

fn bench_first_legal_internal(
    slots: &mut [EnvSlot],
    steps: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let started = Instant::now();
    let mut env_steps = 0_usize;
    for _ in 0..steps {
        slots.par_iter_mut().try_for_each(|slot| {
            let action = first_legal_action(slot)?;
            slot.step_action_index(action)
        })?;
        env_steps += slots.len();
    }
    let elapsed = started.elapsed().as_secs_f64();
    writeln!(out, "OK_BENCH {} {:.12}", env_steps, elapsed).map_err(|err| err.to_string())?;
    out.flush().map_err(|err| err.to_string())
}

fn parse_opt_usize(token: &str) -> Result<Option<usize>, String> {
    let value = token
        .parse::<usize>()
        .map_err(|_| format!("failed to parse usize: {}", token))?;
    if value == 0 {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn main() {
    init_rayon_pool();
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut slots: Vec<EnvSlot> = vec![];
    let pf_particles = std::env::var("AHC061_PF_PARTICLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_PF_PARTICLES);

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                let _ = writeln!(stdout, "ERR stdin {}", err);
                let _ = stdout.flush();
                break;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let result: Result<(), String> = match tokens.first().copied() {
            Some("RESET") => {
                if tokens.len() != 6 {
                    Err("RESET requires: num_envs seed_start seed_stride M_or_0 U_or_0".to_string())
                } else {
                    let num_envs = tokens[1]
                        .parse::<usize>()
                        .map_err(|_| "bad num_envs".to_string());
                    let seed_start = tokens[2]
                        .parse::<u64>()
                        .map_err(|_| "bad seed_start".to_string());
                    let seed_stride = tokens[3]
                        .parse::<u64>()
                        .map_err(|_| "bad seed_stride".to_string());
                    let m_opt = parse_opt_usize(tokens[4]);
                    let u_opt = parse_opt_usize(tokens[5]);
                    match (num_envs, seed_start, seed_stride, m_opt, u_opt) {
                        (Ok(num_envs), Ok(seed_start), Ok(seed_stride), Ok(m_opt), Ok(u_opt)) => {
                            slots = (0..num_envs)
                                .map(|i| {
                                    EnvSlot::from_seed_with_pf(
                                        seed_start + seed_stride * i as u64,
                                        m_opt,
                                        u_opt,
                                        pf_particles,
                                    )
                                })
                                .collect();
                            write_encoded_obs(&slots, &mut stdout).map_err(|err| err.to_string())
                        }
                        _ => Err("failed to parse RESET".to_string()),
                    }
                }
            }
            Some("STEP") => (|| -> Result<(), String> {
                if tokens.len() != slots.len() + 1 {
                    return Err(format!("STEP requires {} actions", slots.len()));
                }
                for (slot, token) in slots.iter_mut().zip(tokens.iter().skip(1)) {
                    let action = token
                        .parse::<usize>()
                        .map_err(|_| format!("bad action: {}", token))?;
                    slot.step_action_index(action)?;
                }
                write_encoded_obs(&slots, &mut stdout).map_err(|err| err.to_string())
            })(),
            Some("STEP_BIN") => (|| -> Result<(), String> {
                if tokens.len() != 1 {
                    return Err("STEP_BIN takes no arguments".to_string());
                }
                let mut actions = vec![0_u8; slots.len()];
                reader
                    .read_exact(&mut actions)
                    .map_err(|err| format!("failed to read STEP_BIN actions: {}", err))?;
                slots
                    .par_iter_mut()
                    .zip(actions.par_iter())
                    .try_for_each(|(slot, &action)| slot.step_action_index(action as usize))?;
                write_encoded_obs(&slots, &mut stdout).map_err(|err| err.to_string())
            })(),
            Some("STEP_FIRST_LEGAL_NOOBS") => {
                if tokens.len() != 1 {
                    Err("STEP_FIRST_LEGAL_NOOBS takes no arguments".to_string())
                } else {
                    step_first_legal_noobs(&mut slots, &mut stdout)
                }
            }
            Some("BENCH_FIRST_LEGAL_INTERNAL") => {
                if tokens.len() != 2 {
                    Err("BENCH_FIRST_LEGAL_INTERNAL requires: steps".to_string())
                } else {
                    match tokens[1].parse::<usize>() {
                        Ok(steps) => bench_first_legal_internal(&mut slots, steps, &mut stdout),
                        Err(_) => Err("bad steps".to_string()),
                    }
                }
            }
            Some("QUIT") => break,
            Some(other) => Err(format!("unknown command: {}", other)),
            None => Ok(()),
        };

        if let Err(err) = result {
            let _ = writeln!(stdout, "ERR {}", err);
            let _ = stdout.flush();
        }
    }
}

fn init_rayon_pool() {
    let Some(threads) = std::env::var("AHC061_ENCODE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
    else {
        return;
    };
    if let Err(err) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        eprintln!(
            "warning: failed to initialize rayon global thread pool: {}",
            err
        );
    }
}
