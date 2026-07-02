use std::io::{self, BufRead, Write};
use std::time::Instant;

use ahc061_rl_tools::vec_env::EnvSlot;

const BOARD_SIZE: usize = 10;
const MAX_PLAYERS: usize = 8;
const MAX_LEVEL: usize = 5;
const ORACLE_PARAMS_PER_PLAYER: usize = 5;
const NUM_PLANES: usize = 77;
const PLANE_M: usize = 24;
const PLANE_U: usize = 25;
const PLANE_SCORE_RATIO: usize = 26;
const PLANE_SCORE_DIFF: usize = 27;
const PLANE_LEGAL_MASK: usize = 28;
const PLANE_PLAYER_SCORE_START: usize = 29;
const PLANE_ORACLE_PARAM_START: usize = PLANE_PLAYER_SCORE_START + MAX_PLAYERS;

fn plane_idx(plane: usize, x: usize, y: usize) -> usize {
    plane * BOARD_SIZE * BOARD_SIZE + x * BOARD_SIZE + y
}

fn fill_plane(planes: &mut [f32], plane: usize, value: f32) {
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            planes[plane_idx(plane, x, y)] = value;
        }
    }
}

fn encode_slot(slot: &EnvSlot) -> (Vec<f32>, Vec<u8>) {
    let mut planes = vec![0.0_f32; NUM_PLANES * BOARD_SIZE * BOARD_SIZE];
    let mut value_sum = 0.0_f32;
    let mut scores = vec![0.0_f32; slot.input.M];

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let value = slot.input.V[x][y] as f32;
            value_sum += value;
            let owner = slot.state.owner[x][y];
            if owner >= 0 {
                scores[owner as usize] += value * slot.state.level[x][y] as f32;
            }
        }
    }
    let mean_value = (value_sum / (BOARD_SIZE * BOARD_SIZE) as f32).max(1.0);
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            planes[plane_idx(0, x, y)] = slot.input.V[x][y] as f32 / mean_value;
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
            planes[plane_idx(mapped_owner, x, y)] = 1.0;
        }
    }

    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            let level = slot.state.level[x][y];
            if (1..=MAX_LEVEL).contains(&level) {
                planes[plane_idx(9 + level, x, y)] = 1.0;
            }
        }
    }

    for player in 0..slot.input.M {
        let mapped_player = player_map[player];
        let (x, y) = slot.state.pos[player];
        if mapped_player < MAX_PLAYERS && x < BOARD_SIZE && y < BOARD_SIZE {
            planes[plane_idx(15 + mapped_player, x, y)] = 1.0;
        }
    }

    fill_plane(&mut planes, 23, (100.0 - slot.turn as f32) / 100.0);
    fill_plane(
        &mut planes,
        PLANE_M,
        slot.input.M as f32 / MAX_PLAYERS as f32,
    );
    fill_plane(&mut planes, PLANE_U, slot.input.U as f32 / MAX_LEVEL as f32);

    let player0_score = scores[0];
    let max_ai_score = scores
        .iter()
        .skip(1)
        .fold(0.0_f32, |acc, &score| acc.max(score));
    let total_capacity = (value_sum * slot.input.U.max(1) as f32).max(1.0);
    fill_plane(
        &mut planes,
        PLANE_SCORE_RATIO,
        player0_score / max_ai_score.max(1.0),
    );
    fill_plane(
        &mut planes,
        PLANE_SCORE_DIFF,
        (player0_score - max_ai_score) / total_capacity,
    );

    let mask = slot
        .mask()
        .into_iter()
        .map(|ok| if ok { 1_u8 } else { 0_u8 })
        .collect::<Vec<_>>();
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            planes[plane_idx(PLANE_LEGAL_MASK, x, y)] = mask[x * BOARD_SIZE + y] as f32;
        }
    }

    for player in 0..slot.input.M {
        let mapped_player = player_map[player];
        if mapped_player < MAX_PLAYERS {
            fill_plane(
                &mut planes,
                PLANE_PLAYER_SCORE_START + mapped_player,
                scores[player] / total_capacity,
            );
            let param_start = PLANE_ORACLE_PARAM_START + mapped_player * ORACLE_PARAMS_PER_PLAYER;
            for param_idx in 0..ORACLE_PARAMS_PER_PLAYER {
                let value = if player > 0 {
                    let ai = player - 1;
                    (match param_idx {
                        0 => slot.input.wa[ai],
                        1 => slot.input.wb[ai],
                        2 => slot.input.wc[ai],
                        3 => slot.input.wd[ai],
                        4 => slot.input.eps[ai],
                        _ => 0.0,
                    }) as f32
                } else {
                    0.0
                };
                fill_plane(&mut planes, param_start + param_idx, value);
            }
        }
    }

    (planes, mask)
}

fn write_encoded_obs(slots: &[EnvSlot], out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "OK {} {} {} {}",
        slots.len(),
        NUM_PLANES,
        BOARD_SIZE,
        BOARD_SIZE
    )?;
    let encoded = slots.iter().map(encode_slot).collect::<Vec<_>>();
    for (planes, _) in &encoded {
        for &value in planes {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    for (_, mask) in &encoded {
        out.write_all(mask)?;
    }
    for slot in slots {
        out.write_all(&(slot.reward as f32).to_le_bytes())?;
    }
    for slot in slots {
        out.write_all(&[if slot.done { 1 } else { 0 }])?;
    }
    for slot in slots {
        out.write_all(&slot.score().to_le_bytes())?;
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
    for slot in slots {
        let action = first_legal_action(slot)?;
        slot.step_action_index(action)?;
    }
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
        for slot in slots.iter_mut() {
            let action = first_legal_action(slot)?;
            slot.step_action_index(action)?;
            env_steps += 1;
        }
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
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut slots: Vec<EnvSlot> = vec![];

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                let _ = writeln!(stdout, "ERR stdin {}", err);
                let _ = stdout.flush();
                break;
            }
        };
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
                                    EnvSlot::from_seed(
                                        seed_start + seed_stride * i as u64,
                                        m_opt,
                                        u_opt,
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
