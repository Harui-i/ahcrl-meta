use std::io::{self, BufRead, Write};

use ahc061_rl_tools::vec_env::EnvSlot;

fn print_slot(slot_id: usize, slot: &EnvSlot, out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "ENV {} {} {} {} {} {} {} {:.12}",
        slot_id,
        slot.input.M,
        slot.input.U,
        slot.turn,
        if slot.done { 1 } else { 0 },
        slot.score(),
        slot.input.N,
        slot.reward
    )?;
    write!(out, "VALUES")?;
    for i in 0..slot.input.N {
        for j in 0..slot.input.N {
            write!(out, " {}", slot.input.V[i][j])?;
        }
    }
    writeln!(out)?;
    write!(out, "OWNER")?;
    for i in 0..slot.input.N {
        for j in 0..slot.input.N {
            write!(out, " {}", slot.state.owner[i][j])?;
        }
    }
    writeln!(out)?;
    write!(out, "LEVEL")?;
    for i in 0..slot.input.N {
        for j in 0..slot.input.N {
            write!(out, " {}", slot.state.level[i][j])?;
        }
    }
    writeln!(out)?;
    write!(out, "POS")?;
    for p in 0..8 {
        if p < slot.input.M {
            write!(out, " {} {}", slot.state.pos[p].0, slot.state.pos[p].1)?;
        } else {
            write!(out, " -1 -1")?;
        }
    }
    writeln!(out)?;
    write!(out, "MASK")?;
    for ok in slot.mask() {
        write!(out, " {}", if ok { 1 } else { 0 })?;
    }
    writeln!(out)
}

fn print_all(slots: &[EnvSlot], out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "OK {}", slots.len())?;
    for (idx, slot) in slots.iter().enumerate() {
        print_slot(idx, slot, out)?;
    }
    writeln!(out, "END")?;
    out.flush()
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
                            print_all(&slots, &mut stdout).map_err(|err| err.to_string())
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
                print_all(&slots, &mut stdout).map_err(|err| err.to_string())
            })(),
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
