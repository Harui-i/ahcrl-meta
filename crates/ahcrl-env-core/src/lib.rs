use std::collections::HashSet;
use std::io::{self, BufRead, Write};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    I64,
    U8,
}

impl DType {
    pub const fn item_size(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::I64 => std::mem::size_of::<i64>(),
            Self::U8 => std::mem::size_of::<u8>(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn elements_per_env(&self) -> Result<usize, String> {
        self.shape.iter().try_fold(1_usize, |size, &dimension| {
            if dimension == 0 {
                return Err(format!("tensor {} has a zero dimension", self.name));
            }
            size.checked_mul(dimension)
                .ok_or_else(|| format!("tensor {} is too large", self.name))
        })
    }

    pub fn bytes_per_env(&self) -> Result<usize, String> {
        self.elements_per_env()?
            .checked_mul(self.dtype.item_size())
            .ok_or_else(|| format!("tensor {} is too large", self.name))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EnvSpec {
    pub protocol_version: u32,
    pub observations: Vec<TensorSpec>,
    pub metrics: Vec<TensorSpec>,
}

impl EnvSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(format!(
                "unsupported protocol version {}, expected {}",
                self.protocol_version, PROTOCOL_VERSION
            ));
        }
        if self.observations.is_empty() {
            return Err("at least one observation tensor is required".to_owned());
        }
        let mut names = HashSet::new();
        for tensor in self.observations.iter().chain(&self.metrics) {
            if tensor.name.is_empty() {
                return Err("tensor names must not be empty".to_owned());
            }
            if matches!(tensor.name.as_str(), "reward" | "done" | "score") {
                return Err(format!("tensor name {} is reserved", tensor.name));
            }
            if !names.insert(tensor.name.clone()) {
                return Err(format!("duplicate tensor name {}", tensor.name));
            }
            tensor.bytes_per_env()?;
        }
        Ok(())
    }

    pub fn batch_bytes(&self, num_envs: usize) -> Result<usize, String> {
        let tensor_bytes =
            self.observations
                .iter()
                .chain(&self.metrics)
                .try_fold(0_usize, |total, tensor| {
                    let size = tensor
                        .bytes_per_env()?
                        .checked_mul(num_envs)
                        .ok_or_else(|| "batch is too large".to_owned())?;
                    total
                        .checked_add(size)
                        .ok_or_else(|| "batch is too large".to_owned())
                })?;
        let required_bytes = num_envs
            .checked_mul(std::mem::size_of::<f32>() + 1 + std::mem::size_of::<i64>())
            .ok_or_else(|| "batch is too large".to_owned())?;
        tensor_bytes
            .checked_add(required_bytes)
            .ok_or_else(|| "batch is too large".to_owned())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub reward: f32,
    pub done: bool,
    pub score: i64,
}

pub trait ContestEnv: Send {
    fn validate_action(&self, action: u32) -> Result<(), String>;
    /// Returns the values reported immediately after constructing or resetting this environment.
    fn initial_outcome(&self) -> StepOutcome;
    /// Advances the environment once and returns the values reported for this transition.
    fn step(&mut self, action: u32) -> Result<StepOutcome, String>;
    fn write_observation(&self, name: &str, destination: &mut [u8]) -> Result<(), String>;
    fn write_metric(&self, name: &str, destination: &mut [u8]) -> Result<(), String>;

    fn visualizer_data(&self) -> Result<VisualizerData, String> {
        Err("visualizer data is unsupported for this environment".to_owned())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VisualizerData {
    pub input: String,
    pub output: String,
}

pub trait EnvFactory: Send + Sync + Sized {
    type Env: ContestEnv;

    fn from_config(config: Value) -> Result<Self, String>;
    fn spec(&self) -> EnvSpec;
    fn create(&self, seed: u64) -> Result<Self::Env, String>;
}

pub struct VecEnvServer<F: EnvFactory> {
    factory: F,
    spec: EnvSpec,
    num_envs: usize,
    envs: Vec<F::Env>,
    outcomes: Vec<StepOutcome>,
    pool: ThreadPool,
}

impl<F: EnvFactory> VecEnvServer<F> {
    pub fn new(factory: F, num_envs: usize) -> Result<Self, String> {
        Self::new_with_workers(factory, num_envs, 0)
    }

    pub fn new_with_workers(factory: F, num_envs: usize, workers: usize) -> Result<Self, String> {
        if num_envs == 0 {
            return Err("num_envs must be positive".to_owned());
        }
        let spec = factory.spec();
        spec.validate()?;
        spec.batch_bytes(num_envs)?;
        let workers = worker_count(workers, num_envs);
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|error| format!("failed to create environment worker pool: {error}"))?;
        Ok(Self {
            factory,
            spec,
            num_envs,
            envs: Vec::new(),
            outcomes: Vec::new(),
            pool,
        })
    }

    pub fn spec(&self) -> &EnvSpec {
        &self.spec
    }

    pub fn reset_all(&mut self, seed_start: u64, seed_stride: u64) -> Result<(), String> {
        let seeds = (0..self.num_envs)
            .map(|env_id| seed_for(seed_start, seed_stride, env_id))
            .collect::<Result<Vec<_>, _>>()?;
        let replacements = self.pool.install(|| {
            seeds
                .par_iter()
                .enumerate()
                .map(|(env_id, &seed)| {
                    self.factory
                        .create(seed)
                        .map(|env| (env.initial_outcome(), env))
                        .map_err(|error| format!("env {env_id}: {error}"))
                })
                .collect::<Vec<_>>()
        });
        let replacements = collect_ordered(replacements)?;
        self.outcomes = replacements.iter().map(|(outcome, _)| *outcome).collect();
        self.envs = replacements.into_iter().map(|(_, env)| env).collect();
        Ok(())
    }

    pub fn reset_mask(
        &mut self,
        mask: &[u8],
        seed_start: u64,
        seed_stride: u64,
    ) -> Result<(), String> {
        self.require_initialized()?;
        if mask.len() != self.num_envs {
            return Err(format!(
                "reset mask length must be {}, got {}",
                self.num_envs,
                mask.len()
            ));
        }
        if let Some(value) = mask.iter().find(|&&value| value > 1) {
            return Err(format!("reset mask contains invalid byte {value}"));
        }
        let reset_ids = mask
            .iter()
            .enumerate()
            .filter_map(|(env_id, &reset)| (reset != 0).then_some(env_id))
            .collect::<Vec<_>>();
        let seeds = reset_ids
            .iter()
            .map(|&env_id| seed_for(seed_start, seed_stride, env_id))
            .collect::<Result<Vec<_>, _>>()?;
        let replacements = self.pool.install(|| {
            reset_ids
                .par_iter()
                .zip(seeds.par_iter())
                .map(|(&env_id, &seed)| {
                    self.factory
                        .create(seed)
                        .map(|env| (env_id, env.initial_outcome(), env))
                        .map_err(|error| format!("env {env_id}: {error}"))
                })
                .collect::<Vec<_>>()
        });
        let replacements = collect_ordered(replacements)?;
        for (env_id, outcome, replacement) in replacements {
            self.envs[env_id] = replacement;
            self.outcomes[env_id] = outcome;
        }
        Ok(())
    }

    pub fn validate_actions(&mut self, actions: &[u32]) -> Result<(), String> {
        self.require_initialized()?;
        if actions.len() != self.num_envs {
            return Err(format!(
                "action count must be {}, got {}",
                self.num_envs,
                actions.len()
            ));
        }
        let results = self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(actions.par_iter())
                .enumerate()
                .map(|(env_id, (env, &action))| {
                    env.validate_action(action)
                        .map_err(|error| format!("env {env_id}: {error}"))
                })
                .collect::<Vec<_>>()
        });
        collect_ordered(results).map(|_| ())
    }

    pub fn step(&mut self, actions: &[u32]) -> Result<(), String> {
        self.validate_actions(actions)?;
        let results = self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(self.outcomes.par_iter_mut())
                .zip(actions.par_iter())
                .enumerate()
                .map(|(env_id, ((env, outcome), &action))| {
                    env.step(action)
                        .map(|next_outcome| *outcome = next_outcome)
                        .map_err(|error| format!("internal step failure in env {env_id}: {error}"))
                })
                .collect::<Vec<_>>()
        });
        collect_ordered(results).map(|_| ())
    }

    /// Advance only the environments selected by `mask`.
    ///
    /// This is primarily useful for finite evaluation batches: environments that
    /// have already finished can remain in the batch without requiring a reset.
    pub fn step_mask(&mut self, mask: &[u8], actions: &[u32]) -> Result<(), String> {
        self.require_initialized()?;
        if mask.len() != self.num_envs {
            return Err(format!(
                "step mask length must be {}, got {}",
                self.num_envs,
                mask.len()
            ));
        }
        if actions.len() != self.num_envs {
            return Err(format!(
                "action count must be {}, got {}",
                self.num_envs,
                actions.len()
            ));
        }
        if let Some(value) = mask.iter().find(|&&value| value > 1) {
            return Err(format!("step mask contains invalid byte {value}"));
        }
        let validation = self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(mask.par_iter())
                .zip(actions.par_iter())
                .enumerate()
                .map(|(env_id, ((env, &selected), &action))| {
                    (selected == 0).then_some(()).map_or_else(
                        || {
                            env.validate_action(action)
                                .map_err(|error| format!("env {env_id}: {error}"))
                        },
                        Ok,
                    )
                })
                .collect::<Vec<_>>()
        });
        collect_ordered(validation)?;
        let results = self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(self.outcomes.par_iter_mut())
                .zip(mask.par_iter())
                .zip(actions.par_iter())
                .enumerate()
                .map(|(env_id, (((env, outcome), &selected), &action))| {
                    if selected != 0 {
                        env.step(action)
                            .map(|next_outcome| *outcome = next_outcome)
                            .map_err(|error| {
                                format!("internal step failure in env {env_id}: {error}")
                            })
                    } else {
                        Ok(())
                    }
                })
                .collect::<Vec<_>>()
        });
        collect_ordered(results).map(|_| ())
    }

    pub fn visualizer_data(&self) -> Result<Vec<VisualizerData>, String> {
        self.require_initialized()?;
        self.envs
            .iter()
            .enumerate()
            .map(|(env_id, env)| {
                env.visualizer_data()
                    .map_err(|error| format!("env {env_id}: {error}"))
            })
            .collect()
    }

    pub fn encode_batch(&mut self) -> Result<Vec<u8>, String> {
        self.require_initialized()?;
        let capacity = self.spec.batch_bytes(self.num_envs)?;
        let mut output = Vec::with_capacity(capacity);
        for tensor in self.spec.observations.clone() {
            self.encode_tensor(&tensor, false, &mut output)?;
        }
        self.encode_outcomes(&mut output);
        for tensor in self.spec.metrics.clone() {
            self.encode_tensor(&tensor, true, &mut output)?;
        }
        debug_assert_eq!(output.len(), capacity);
        Ok(output)
    }

    fn encode_tensor(
        &mut self,
        tensor: &TensorSpec,
        metric: bool,
        output: &mut Vec<u8>,
    ) -> Result<(), String> {
        let bytes_per_env = tensor.bytes_per_env()?;
        let start = output.len();
        output.resize(start + bytes_per_env * self.num_envs, 0);
        let results = self.pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(output[start..].par_chunks_mut(bytes_per_env))
                .enumerate()
                .map(|(env_id, (env, destination))| {
                    let result = if metric {
                        env.write_metric(&tensor.name, destination)
                    } else {
                        env.write_observation(&tensor.name, destination)
                    };
                    result.map_err(|error| {
                        format!("failed to encode {} for env {env_id}: {error}", tensor.name)
                    })
                })
                .collect::<Vec<_>>()
        });
        collect_ordered(results).map(|_| ())
    }

    fn encode_outcomes(&self, output: &mut Vec<u8>) {
        let reward_start = output.len();
        output.resize(reward_start + self.num_envs * std::mem::size_of::<f32>(), 0);
        let outcomes = &self.outcomes;
        self.pool.install(|| {
            output[reward_start..]
                .par_chunks_mut(std::mem::size_of::<f32>())
                .zip(outcomes.par_iter())
                .for_each(|(destination, outcome)| {
                    destination.copy_from_slice(&outcome.reward.to_le_bytes())
                });
        });
        let done_start = output.len();
        output.resize(done_start + self.num_envs, 0);
        self.pool.install(|| {
            output[done_start..]
                .par_iter_mut()
                .zip(outcomes.par_iter())
                .for_each(|(destination, outcome)| *destination = u8::from(outcome.done));
        });
        let score_start = output.len();
        output.resize(score_start + self.num_envs * std::mem::size_of::<i64>(), 0);
        self.pool.install(|| {
            output[score_start..]
                .par_chunks_mut(std::mem::size_of::<i64>())
                .zip(outcomes.par_iter())
                .for_each(|(destination, outcome)| {
                    destination.copy_from_slice(&outcome.score.to_le_bytes())
                });
        });
    }

    fn require_initialized(&self) -> Result<(), String> {
        if self.envs.len() != self.num_envs {
            Err("environment has not been reset".to_owned())
        } else {
            Ok(())
        }
    }
}

fn worker_count(requested_workers: usize, num_envs: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let desired = if requested_workers == 0 {
        available
    } else {
        requested_workers
    };
    desired.min(num_envs).max(1)
}

fn collect_ordered<T>(results: Vec<Result<T, String>>) -> Result<Vec<T>, String> {
    results.into_iter().collect()
}

fn seed_for(seed_start: u64, seed_stride: u64, env_id: usize) -> Result<u64, String> {
    let offset = seed_stride
        .checked_mul(env_id as u64)
        .ok_or_else(|| "seed overflow".to_owned())?;
    seed_start
        .checked_add(offset)
        .ok_or_else(|| "seed overflow".to_owned())
}

#[derive(Deserialize)]
struct InitRequest {
    protocol_version: u32,
    num_envs: usize,
    env_workers: usize,
    config: Value,
}

pub fn run_server<F: EnvFactory>() -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_server_with_io::<F, _, _>(stdin.lock(), stdout.lock())
}

pub fn server_main<F: EnvFactory>() {
    if let Err(error) = run_server::<F>() {
        eprintln!("ahcrl env server failed: {error}");
        std::process::exit(1);
    }
}

pub fn run_server_with_io<F, R, W>(mut reader: R, mut writer: W) -> Result<(), String>
where
    F: EnvFactory,
    R: BufRead,
    W: Write,
{
    let mut server: Option<VecEnvServer<F>> = None;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| format!("failed to read command: {error}"))?;
        if read == 0 {
            return if server.is_some() {
                Err("stdin closed before QUIT".to_owned())
            } else {
                Err("stdin closed before INIT".to_owned())
            };
        }
        let line = line.trim_end_matches(['\r', '\n']);
        let mut parts = line.split_whitespace();
        let command = parts.next().unwrap_or_default();
        match command {
            "INIT" => {
                if server.is_some() {
                    send_error(&mut writer, "INIT may only be sent once")?;
                    continue;
                }
                let Some(length) = parse_single_usize(parts) else {
                    send_error(&mut writer, "INIT requires one JSON byte length")?;
                    continue;
                };
                let mut json = vec![0_u8; length];
                reader
                    .read_exact(&mut json)
                    .map_err(|error| format!("failed to read INIT payload: {error}"))?;
                let request: InitRequest = match serde_json::from_slice(&json) {
                    Ok(request) => request,
                    Err(error) => {
                        send_error(&mut writer, &format!("invalid INIT JSON: {error}"))?;
                        continue;
                    }
                };
                if request.protocol_version != PROTOCOL_VERSION {
                    send_error(
                        &mut writer,
                        &format!(
                            "unsupported protocol version {}, expected {}",
                            request.protocol_version, PROTOCOL_VERSION
                        ),
                    )?;
                    continue;
                }
                let factory = match F::from_config(request.config) {
                    Ok(factory) => factory,
                    Err(error) => {
                        send_error(&mut writer, &format!("invalid config: {error}"))?;
                        continue;
                    }
                };
                let new_server = match VecEnvServer::new_with_workers(
                    factory,
                    request.num_envs,
                    request.env_workers,
                ) {
                    Ok(server) => server,
                    Err(error) => {
                        send_error(&mut writer, &error)?;
                        continue;
                    }
                };
                let spec_json = serde_json::to_vec(new_server.spec())
                    .map_err(|error| format!("failed to serialize schema: {error}"))?;
                writeln!(writer, "OK_SPEC {}", spec_json.len())
                    .map_err(|error| format!("failed to write schema header: {error}"))?;
                writer
                    .write_all(&spec_json)
                    .map_err(|error| format!("failed to write schema: {error}"))?;
                writer
                    .write_all(b"\n")
                    .map_err(|error| format!("failed to terminate schema: {error}"))?;
                writer
                    .flush()
                    .map_err(|error| format!("failed to flush schema: {error}"))?;
                server = Some(new_server);
            }
            "RESET_ALL" => {
                let Some((seed_start, seed_stride)) = parse_two_u64(parts) else {
                    send_error(&mut writer, "RESET_ALL requires seed_start and seed_stride")?;
                    continue;
                };
                let Some(server) = server.as_mut() else {
                    send_error(&mut writer, "INIT must be sent first")?;
                    continue;
                };
                if let Err(error) = server.reset_all(seed_start, seed_stride) {
                    send_error(&mut writer, &error)?;
                    continue;
                }
                write_batch(server, &mut writer)?;
            }
            "RESET_MASK" => {
                let Some((seed_start, seed_stride)) = parse_two_u64(parts) else {
                    send_error(
                        &mut writer,
                        "RESET_MASK requires seed_start and seed_stride",
                    )?;
                    continue;
                };
                let Some(server) = server.as_mut() else {
                    send_error(&mut writer, "INIT must be sent first")?;
                    continue;
                };
                let mut mask = vec![0_u8; server.num_envs];
                reader
                    .read_exact(&mut mask)
                    .map_err(|error| format!("failed to read reset mask: {error}"))?;
                if let Err(error) = server.reset_mask(&mask, seed_start, seed_stride) {
                    send_error(&mut writer, &error)?;
                    continue;
                }
                write_batch(server, &mut writer)?;
            }
            "STEP" => {
                if parts.next().is_some() {
                    send_error(&mut writer, "STEP takes no arguments")?;
                    continue;
                }
                let Some(server) = server.as_mut() else {
                    send_error(&mut writer, "INIT must be sent first")?;
                    continue;
                };
                let mut bytes = vec![0_u8; server.num_envs * std::mem::size_of::<u32>()];
                reader
                    .read_exact(&mut bytes)
                    .map_err(|error| format!("failed to read actions: {error}"))?;
                let (action_bytes, remainder) = bytes.as_chunks::<4>();
                debug_assert!(remainder.is_empty());
                let actions = action_bytes
                    .iter()
                    .map(|&chunk| u32::from_le_bytes(chunk))
                    .collect::<Vec<_>>();
                if let Err(error) = server.validate_actions(&actions) {
                    send_error(&mut writer, &error)?;
                    continue;
                }
                server.step(&actions)?;
                write_batch(server, &mut writer)?;
            }
            "STEP_MASK" => {
                if parts.next().is_some() {
                    send_error(&mut writer, "STEP_MASK takes no arguments")?;
                    continue;
                }
                let Some(server) = server.as_mut() else {
                    send_error(&mut writer, "INIT must be sent first")?;
                    continue;
                };
                let mut mask = vec![0_u8; server.num_envs];
                reader
                    .read_exact(&mut mask)
                    .map_err(|error| format!("failed to read step mask: {error}"))?;
                let mut bytes = vec![0_u8; server.num_envs * std::mem::size_of::<u32>()];
                reader
                    .read_exact(&mut bytes)
                    .map_err(|error| format!("failed to read actions: {error}"))?;
                let (action_bytes, remainder) = bytes.as_chunks::<4>();
                debug_assert!(remainder.is_empty());
                let actions = action_bytes
                    .iter()
                    .map(|&chunk| u32::from_le_bytes(chunk))
                    .collect::<Vec<_>>();
                if let Err(error) = server.step_mask(&mask, &actions) {
                    send_error(&mut writer, &error)?;
                    continue;
                }
                write_batch(server, &mut writer)?;
            }
            "VISUALIZER_DATA" => {
                if parts.next().is_some() {
                    send_error(&mut writer, "VISUALIZER_DATA takes no arguments")?;
                    continue;
                }
                let Some(server) = server.as_ref() else {
                    send_error(&mut writer, "INIT must be sent first")?;
                    continue;
                };
                let data = match server.visualizer_data() {
                    Ok(data) => data,
                    Err(error) => {
                        send_error(&mut writer, &error)?;
                        continue;
                    }
                };
                let payload = serde_json::to_vec(&data)
                    .map_err(|error| format!("failed to serialize visualizer data: {error}"))?;
                writeln!(writer, "OK_VISUALIZER_DATA {}", payload.len())
                    .map_err(|error| format!("failed to write visualizer data header: {error}"))?;
                writer
                    .write_all(&payload)
                    .and_then(|()| writer.write_all(b"\n"))
                    .and_then(|()| writer.flush())
                    .map_err(|error| format!("failed to write visualizer data: {error}"))?;
            }
            "QUIT" => {
                if parts.next().is_some() {
                    send_error(&mut writer, "QUIT takes no arguments")?;
                    continue;
                }
                writer
                    .write_all(b"OK_QUIT\n")
                    .map_err(|error| format!("failed to write QUIT response: {error}"))?;
                writer
                    .flush()
                    .map_err(|error| format!("failed to flush QUIT response: {error}"))?;
                return Ok(());
            }
            _ => {
                send_error(&mut writer, &format!("unknown command {command:?}"))?;
            }
        }
    }
}

fn parse_single_usize<'a>(mut parts: impl Iterator<Item = &'a str>) -> Option<usize> {
    let value = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(value)
}

fn parse_two_u64<'a>(mut parts: impl Iterator<Item = &'a str>) -> Option<(u64, u64)> {
    let first = parts.next()?.parse().ok()?;
    let second = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((first, second))
}

fn send_error(writer: &mut impl Write, error: &str) -> Result<(), String> {
    let message = error.replace(['\r', '\n'], " ");
    writeln!(writer, "ERR {message}")
        .and_then(|()| writer.flush())
        .map_err(|io_error| format!("failed to write error response: {io_error}"))
}

fn write_batch<F: EnvFactory>(
    server: &mut VecEnvServer<F>,
    writer: &mut impl Write,
) -> Result<(), String> {
    let batch = server.encode_batch()?;
    writeln!(writer, "OK_BATCH {}", batch.len())
        .map_err(|error| format!("failed to write batch header: {error}"))?;
    writer
        .write_all(&batch)
        .map_err(|error| format!("failed to write batch: {error}"))?;
    writer
        .write_all(b"\nEND\n")
        .and_then(|()| writer.flush())
        .map_err(|error| format!("failed to finish batch: {error}"))
}

pub fn write_f32_slice(values: &[f32], destination: &mut [u8]) -> Result<(), String> {
    write_numeric_slice(values, destination, f32::to_le_bytes)
}

pub fn write_i64_slice(values: &[i64], destination: &mut [u8]) -> Result<(), String> {
    write_numeric_slice(values, destination, i64::to_le_bytes)
}

fn write_numeric_slice<T, const N: usize>(
    values: &[T],
    destination: &mut [u8],
    to_bytes: impl Fn(T) -> [u8; N],
) -> Result<(), String>
where
    T: Copy,
{
    let expected = values
        .len()
        .checked_mul(N)
        .ok_or_else(|| "numeric slice is too large".to_owned())?;
    if destination.len() != expected {
        return Err(format!(
            "destination has {} bytes, expected {expected}",
            destination.len()
        ));
    }
    let (chunks, remainder) = destination.as_chunks_mut::<N>();
    debug_assert!(remainder.is_empty());
    for (&value, chunk) in values.iter().zip(chunks) {
        chunk.copy_from_slice(&to_bytes(value));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DummyEnv {
        value: u32,
    }

    impl ContestEnv for DummyEnv {
        fn validate_action(&self, action: u32) -> Result<(), String> {
            if action <= 2 {
                Ok(())
            } else {
                Err(format!("invalid action {action}"))
            }
        }

        fn initial_outcome(&self) -> StepOutcome {
            StepOutcome {
                reward: 0.0,
                done: false,
                score: self.value as i64,
            }
        }

        fn step(&mut self, action: u32) -> Result<StepOutcome, String> {
            self.value += action;
            Ok(StepOutcome {
                reward: action as f32,
                done: false,
                score: self.value as i64,
            })
        }

        fn write_observation(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
            if name != "value" || destination.len() != 4 {
                return Err("unexpected observation".to_owned());
            }
            destination.copy_from_slice(&(self.value as f32).to_le_bytes());
            Ok(())
        }

        fn write_metric(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
            if name != "double" || destination.len() != 8 {
                return Err("unexpected metric".to_owned());
            }
            destination.copy_from_slice(&(2 * self.value as i64).to_le_bytes());
            Ok(())
        }
    }

    struct DummyFactory;

    impl EnvFactory for DummyFactory {
        type Env = DummyEnv;

        fn from_config(_config: Value) -> Result<Self, String> {
            Ok(Self)
        }

        fn spec(&self) -> EnvSpec {
            EnvSpec {
                protocol_version: PROTOCOL_VERSION,
                observations: vec![TensorSpec {
                    name: "value".to_owned(),
                    dtype: DType::F32,
                    shape: vec![1],
                }],
                metrics: vec![TensorSpec {
                    name: "double".to_owned(),
                    dtype: DType::I64,
                    shape: vec![1],
                }],
            }
        }

        fn create(&self, seed: u64) -> Result<Self::Env, String> {
            Ok(DummyEnv { value: seed as u32 })
        }
    }

    #[derive(Clone)]
    struct ProbeEnv {
        inner: DummyEnv,
        in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ContestEnv for ProbeEnv {
        fn validate_action(&self, action: u32) -> Result<(), String> {
            self.inner.validate_action(action)
        }

        fn initial_outcome(&self) -> StepOutcome {
            self.inner.initial_outcome()
        }

        fn step(&mut self, action: u32) -> Result<StepOutcome, String> {
            let in_flight = self
                .in_flight
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.max_in_flight
                .fetch_max(in_flight, std::sync::atomic::Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(10));
            self.in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.step(action)
        }

        fn write_observation(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
            self.inner.write_observation(name, destination)
        }

        fn write_metric(&self, name: &str, destination: &mut [u8]) -> Result<(), String> {
            self.inner.write_metric(name, destination)
        }
    }

    struct ProbeFactory {
        in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ProbeFactory {
        fn new() -> Self {
            Self {
                in_flight: Default::default(),
                max_in_flight: Default::default(),
            }
        }
    }

    impl EnvFactory for ProbeFactory {
        type Env = ProbeEnv;

        fn from_config(_config: Value) -> Result<Self, String> {
            Ok(Self::new())
        }

        fn spec(&self) -> EnvSpec {
            DummyFactory.spec()
        }

        fn create(&self, seed: u64) -> Result<Self::Env, String> {
            Ok(ProbeEnv {
                inner: DummyEnv { value: seed as u32 },
                in_flight: self.in_flight.clone(),
                max_in_flight: self.max_in_flight.clone(),
            })
        }
    }

    #[test]
    fn reset_mask_preserves_unselected_slots() {
        let mut server = VecEnvServer::new(DummyFactory, 3).unwrap();
        server.reset_all(10, 2).unwrap();
        server.reset_mask(&[0, 1, 0], 100, 3).unwrap();
        assert_eq!(server.envs[0].value, 10);
        assert_eq!(server.envs[1].value, 103);
        assert_eq!(server.envs[2].value, 14);
        assert_eq!(
            server.outcomes,
            vec![
                StepOutcome {
                    reward: 0.0,
                    done: false,
                    score: 10
                },
                StepOutcome {
                    reward: 0.0,
                    done: false,
                    score: 103
                },
                StepOutcome {
                    reward: 0.0,
                    done: false,
                    score: 14
                },
            ]
        );
    }

    #[test]
    fn invalid_batch_does_not_mutate_any_slot() {
        let mut server = VecEnvServer::new(DummyFactory, 2).unwrap();
        server.reset_all(5, 1).unwrap();
        assert!(server.step(&[1, 9]).is_err());
        assert_eq!(server.envs[0].value, 5);
        assert_eq!(server.envs[1].value, 6);
    }

    #[test]
    fn step_mask_skips_unselected_slots_without_validating_their_actions() {
        let mut server = VecEnvServer::new(DummyFactory, 2).unwrap();
        server.reset_all(5, 1).unwrap();
        server.step_mask(&[1, 0], &[2, 99]).unwrap();
        assert_eq!(server.envs[0].value, 7);
        assert_eq!(server.envs[1].value, 6);
        assert_eq!(
            server.outcomes,
            vec![
                StepOutcome {
                    reward: 2.0,
                    done: false,
                    score: 7
                },
                StepOutcome {
                    reward: 0.0,
                    done: false,
                    score: 6
                },
            ]
        );
    }

    #[test]
    fn schema_rejects_duplicate_and_reserved_names() {
        let duplicate = EnvSpec {
            protocol_version: PROTOCOL_VERSION,
            observations: vec![TensorSpec {
                name: "same".to_owned(),
                dtype: DType::U8,
                shape: vec![1],
            }],
            metrics: vec![TensorSpec {
                name: "same".to_owned(),
                dtype: DType::U8,
                shape: vec![1],
            }],
        };
        assert!(duplicate.validate().is_err());

        let reserved = EnvSpec {
            protocol_version: PROTOCOL_VERSION,
            observations: vec![TensorSpec {
                name: "reward".to_owned(),
                dtype: DType::F32,
                shape: vec![1],
            }],
            metrics: vec![],
        };
        assert!(reserved.validate().is_err());
    }

    #[test]
    fn encoded_batch_uses_declared_field_order_and_little_endian_values() {
        let mut server = VecEnvServer::new(DummyFactory, 2).unwrap();
        server.reset_all(3, 1).unwrap();
        let batch = server.encode_batch().unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&3.0_f32.to_le_bytes());
        expected.extend_from_slice(&4.0_f32.to_le_bytes());
        expected.extend_from_slice(&0.0_f32.to_le_bytes());
        expected.extend_from_slice(&0.0_f32.to_le_bytes());
        expected.extend_from_slice(&[0, 0]);
        expected.extend_from_slice(&3_i64.to_le_bytes());
        expected.extend_from_slice(&4_i64.to_le_bytes());
        expected.extend_from_slice(&6_i64.to_le_bytes());
        expected.extend_from_slice(&8_i64.to_le_bytes());
        assert_eq!(batch, expected);
    }

    #[test]
    fn parallel_workers_preserve_slot_order_and_batch_bytes() {
        let mut serial = VecEnvServer::new_with_workers(DummyFactory, 4, 1).unwrap();
        let mut parallel = VecEnvServer::new_with_workers(DummyFactory, 4, 4).unwrap();
        serial.reset_all(10, 3).unwrap();
        parallel.reset_all(10, 3).unwrap();
        let actions = [1, 2, 1, 2];
        serial.step(&actions).unwrap();
        parallel.step(&actions).unwrap();

        assert_eq!(serial.outcomes, parallel.outcomes);
        assert_eq!(
            serial.encode_batch().unwrap(),
            parallel.encode_batch().unwrap()
        );
    }

    #[test]
    fn worker_count_controls_step_concurrency() {
        let serial_factory = ProbeFactory::new();
        let serial_max = serial_factory.max_in_flight.clone();
        let mut serial = VecEnvServer::new_with_workers(serial_factory, 4, 1).unwrap();
        serial.reset_all(0, 1).unwrap();
        serial.step(&[1, 1, 1, 1]).unwrap();
        assert_eq!(serial_max.load(std::sync::atomic::Ordering::SeqCst), 1);

        let parallel_factory = ProbeFactory::new();
        let parallel_max = parallel_factory.max_in_flight.clone();
        let mut parallel = VecEnvServer::new_with_workers(parallel_factory, 4, 4).unwrap();
        parallel.reset_all(0, 1).unwrap();
        parallel.step(&[1, 1, 1, 1]).unwrap();
        assert!(parallel_max.load(std::sync::atomic::Ordering::SeqCst) > 1);
    }

    #[test]
    fn protocol_handles_init_reset_step_and_quit() {
        let init = serde_json::json!({
            "protocol_version": PROTOCOL_VERSION,
            "num_envs": 2,
            "env_workers": 1,
            "config": {},
        });
        let init = serde_json::to_vec(&init).unwrap();
        let mut input = format!("INIT {}\n", init.len()).into_bytes();
        input.extend_from_slice(&init);
        input.extend_from_slice(b"RESET_ALL 7 1\nSTEP\n");
        input.extend_from_slice(&1_u32.to_le_bytes());
        input.extend_from_slice(&2_u32.to_le_bytes());
        input.extend_from_slice(b"QUIT\n");
        let mut output = Vec::new();
        run_server_with_io::<DummyFactory, _, _>(io::Cursor::new(input), &mut output).unwrap();
        let output = String::from_utf8_lossy(&output);
        assert!(output.starts_with("OK_SPEC "));
        assert_eq!(output.matches("OK_BATCH ").count(), 2);
        assert!(output.ends_with("OK_QUIT\n"));
    }

    #[test]
    fn protocol_rejects_truncated_action_payload() {
        let init = serde_json::json!({
            "protocol_version": PROTOCOL_VERSION,
            "num_envs": 2,
            "env_workers": 1,
            "config": {},
        });
        let init = serde_json::to_vec(&init).unwrap();
        let mut input = format!("INIT {}\n", init.len()).into_bytes();
        input.extend_from_slice(&init);
        input.extend_from_slice(b"RESET_ALL 7 1\nSTEP\n");
        input.extend_from_slice(&1_u32.to_le_bytes());
        let error =
            run_server_with_io::<DummyFactory, _, _>(io::Cursor::new(input), &mut Vec::new())
                .unwrap_err();
        assert!(error.contains("failed to read actions"));
    }
}
