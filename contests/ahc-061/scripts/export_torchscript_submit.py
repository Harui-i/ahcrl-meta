from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

from ahcrl.contests.ahc061.encoder import NUM_PLANES
from ahcrl.contests.ahc061.model import ActorCritic, RunningObservationNormalizer

BASE91_ALPHABET = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~"'
)


def base91_encode(data: bytes) -> str:
    out: list[str] = []
    bit_queue = 0
    bit_count = 0
    for byte in data:
        bit_queue |= byte << bit_count
        bit_count += 8
        if bit_count > 13:
            value = bit_queue & 8191
            if value > 88:
                bit_queue >>= 13
                bit_count -= 13
            else:
                value = bit_queue & 16383
                bit_queue >>= 14
                bit_count -= 14
            out.append(BASE91_ALPHABET[value % 91])
            out.append(BASE91_ALPHABET[value // 91])
    if bit_count:
        out.append(BASE91_ALPHABET[bit_queue % 91])
        if bit_count > 7 or bit_queue > 90:
            out.append(BASE91_ALPHABET[bit_queue // 91])
    return "".join(out)


def c_string_literal_chunks(s: str, *, width: int = 120) -> str:
    chunks = []
    for start in range(0, len(s), width):
        chunk = s[start : start + width]
        chunk = chunk.replace("\\", "\\\\").replace('"', '\\"')
        chunks.append(f'    "{chunk}"')
    return "\n".join(chunks)


def load_export_model(checkpoint_path: Path, config: dict[str, object]) -> ActorCritic:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or "model" not in checkpoint:
        raise ValueError("checkpoint must use the shared training checkpoint format")
    model = ActorCritic(
        channels=int(cast(Any, config["model_channels"])),
        blocks=int(cast(Any, config["model_blocks"])),
        block_type=str(config["model_block_type"]),
    ).to(dtype=torch.bfloat16)
    if bool(config.get("obs_norm", True)):
        model.observation_normalizer = RunningObservationNormalizer(
            NUM_PLANES,
            float(cast(Any, config.get("obs_norm_epsilon", 1e-8))),
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def pack_q4_policy(checkpoint_path: Path, config: dict[str, object]) -> bytes:
    """Pack the actor-only network as symmetric int4, with fp16 group scales.

    The value head is intentionally omitted: the submission only consumes policy
    logits.  Small tensors stay fp16, while larger tensors use 128-value groups.
    """
    model = load_export_model(checkpoint_path, config)
    state_dict = model.state_dict()
    tensors = [
        value.detach().float().contiguous()
        for name, value in state_dict.items()
        if name.startswith("trunk.") or name.startswith("policy.")
    ]
    obs_normalizer = model.observation_normalizer
    if obs_normalizer is None:
        raise ValueError("q4 submit requires observation normalization")
    mean = obs_normalizer.mean.detach().float().contiguous()
    count = obs_normalizer.count.to(dtype=obs_normalizer.m2.dtype).clamp_min(1)
    variance = torch.where(
        obs_normalizer.count > 0,
        obs_normalizer.m2 / count,
        torch.ones_like(obs_normalizer.m2),
    )
    invstd = torch.rsqrt(variance + obs_normalizer.epsilon).contiguous()

    packed = bytearray(b"AHC061Q4\x01")
    packed += struct.pack("<H", len(tensors))
    for tensor in tensors:
        values = tensor.reshape(-1)
        n = values.numel()
        if n < 2048:
            packed += b"\x01" + struct.pack("<I", n)
            packed += values.to(torch.float16).numpy().tobytes()
            continue
        packed += b"\x00" + struct.pack("<I", n)
        codes = bytearray()
        scales = bytearray()
        for start in range(0, n, 128):
            group = values[start : start + 128]
            scale = float(group.abs().max()) / 7.0
            if scale == 0.0:
                scale = 1.0
            scales += struct.pack("<e", scale)
            quantized = torch.clamp(torch.round(group / scale), -8, 7).to(torch.int8)
            for index in range(0, quantized.numel(), 2):
                lo = int(quantized[index]) & 15
                hi = int(quantized[index + 1]) & 15 if index + 1 < quantized.numel() else 0
                codes.append(lo | (hi << 4))
        packed += codes + scales
    for tensor in (mean, invstd):
        values = tensor.reshape(-1)
        packed += b"\x01" + struct.pack("<I", values.numel())
        packed += values.to(torch.float16).numpy().tobytes()
    return bytes(packed)


def export_torchscript(checkpoint_path: Path, config: dict[str, object]) -> bytes:
    model = load_export_model(checkpoint_path, config)
    dummy = torch.zeros((1, NUM_PLANES, 10, 10), dtype=torch.bfloat16)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=True)
        frozen = torch.jit.freeze(traced)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        frozen.save(f.name)
        return Path(f.name).read_bytes()


def render_cpp(
    encoded_model: str,
    *,
    checkpoint_name: str,
    torchscript_size: int,
    pf_particles: int,
    temperature: float,
) -> str:
    encoded_chunks = c_string_literal_chunks(encoded_model)
    return f"""#include <ATen/Parallel.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {{

constexpr int N = 10;
constexpr int T = 100;
constexpr int MAX_PLAYERS = 8;
constexpr int MAX_LEVEL = 5;
constexpr int NUM_PLANES = {NUM_PLANES};
constexpr int PF_PARTICLES = {pf_particles};
constexpr double ACTION_TEMPERATURE = {temperature:.17g};
constexpr int ORACLE_PARAMS_PER_PLAYER = 5;
constexpr int PLAYER_AGG_FEATURES = 4;
constexpr int PLANE_M = 24;
constexpr int PLANE_U = 25;
constexpr int PLANE_SCORE_RATIO = 26;
constexpr int PLANE_SCORE_DIFF = 27;
constexpr int PLANE_LEGAL_MASK = 28;
constexpr int PLANE_PLAYER_SCORE_START = 29;
constexpr int PLANE_ORACLE_PARAM_START = PLANE_PLAYER_SCORE_START + MAX_PLAYERS;
constexpr int PLANE_COMP_START = PLANE_ORACLE_PARAM_START + MAX_PLAYERS * ORACLE_PARAMS_PER_PLAYER;
constexpr int PLANE_REACH_START = PLANE_COMP_START + MAX_PLAYERS;
constexpr int PLANE_NEXT_GREEDY_START = PLANE_REACH_START + MAX_PLAYERS;
constexpr int PLANE_DIST_OWNER_START = PLANE_NEXT_GREEDY_START + MAX_PLAYERS;
constexpr int PLANE_DIST_COMP_START = PLANE_DIST_OWNER_START + MAX_PLAYERS;
constexpr int PLANE_DIST_CENTER = PLANE_DIST_COMP_START + MAX_PLAYERS;
constexpr int PLANE_X_NORM = PLANE_DIST_CENTER + 1;
constexpr int PLANE_Y_NORM = PLANE_X_NORM + 1;
constexpr int PLANE_POS0_X_NORM = PLANE_Y_NORM + 1;
constexpr int PLANE_POS0_Y_NORM = PLANE_POS0_X_NORM + 1;
constexpr int PLANE_PLAYER_AGG_START = PLANE_POS0_Y_NORM + 1;
constexpr int PLAYER_AGG_OWNER_LEVEL_SUM = 0;
constexpr int PLAYER_AGG_OWNER_LEVEL_VALUE_SUM = 1;
constexpr int PLAYER_AGG_COMP_LEVEL_SUM = 2;
constexpr int PLAYER_AGG_COMP_LEVEL_VALUE_SUM = 3;

const string kBase91Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{{|}}~\\\"";

const string kEncodedModel =
{encoded_chunks};

vector<unsigned char> decode_base91(const string& input) {{
    array<int, 256> decode;
    decode.fill(-1);
    for (int i = 0; i < 91; ++i) {{
        decode[static_cast<unsigned char>(kBase91Alphabet[i])] = i;
    }}

    vector<unsigned char> out;
    int value = -1;
    unsigned int bit_queue = 0;
    int bit_count = 0;
    for (unsigned char ch : input) {{
        int decoded = decode[ch];
        if (decoded < 0) continue;
        if (value < 0) {{
            value = decoded;
        }} else {{
            value += decoded * 91;
            bit_queue |= static_cast<unsigned int>(value) << bit_count;
            bit_count += (value & 8191) > 88 ? 13 : 14;
            do {{
                out.push_back(static_cast<unsigned char>(bit_queue & 255));
                bit_queue >>= 8;
                bit_count -= 8;
            }} while (bit_count > 7);
            value = -1;
        }}
    }}
    if (value >= 0) {{
        out.push_back(static_cast<unsigned char>((bit_queue | (value << bit_count)) & 255));
    }}
    return out;
}}

struct State {{
    int m = 0;
    int u = 0;
    int turn = 0;
    int values[N][N]{{}};
    int owner[N][N]{{}};
    int level[N][N]{{}};
    pair<int, int> pos[MAX_PLAYERS]{{}};
}};

vector<pair<int, int>> get_candidates(const State& st, int player) {{
    vector<pair<int, int>> reachable;
    bool visited[N][N]{{}};
    queue<pair<int, int>> q;
    q.push(st.pos[player]);
    visited[st.pos[player].first][st.pos[player].second] = true;

    constexpr int dx[4] = {{0, 1, 0, -1}};
    constexpr int dy[4] = {{1, 0, -1, 0}};
    while (!q.empty()) {{
        auto [x, y] = q.front();
        q.pop();
        bool ok = true;
        for (int p = 0; p < st.m; ++p) {{
            if (p != player && st.pos[p] == make_pair(x, y)) {{
                ok = false;
                break;
            }}
        }}
        if (ok) reachable.push_back({{x, y}});
        if (st.owner[x][y] != player) continue;
        for (int d = 0; d < 4; ++d) {{
            int nx = x + dx[d];
            int ny = y + dy[d];
            if (0 <= nx && nx < N && 0 <= ny && ny < N && !visited[nx][ny]) {{
                visited[nx][ny] = true;
                q.push({{nx, ny}});
            }}
        }}
    }}
    return reachable;
}}

array<unsigned char, N * N> connected_component_mask(const State& st, int player) {{
    array<unsigned char, N * N> mask{{}};
    auto [sx, sy] = st.pos[player];
    if (st.owner[sx][sy] != player) return mask;

    queue<pair<int, int>> q;
    mask[sx * N + sy] = 1;
    q.push({{sx, sy}});
    constexpr int dx[4] = {{0, 1, 0, -1}};
    constexpr int dy[4] = {{1, 0, -1, 0}};
    while (!q.empty()) {{
        auto [x, y] = q.front();
        q.pop();
        for (int d = 0; d < 4; ++d) {{
            int nx = x + dx[d];
            int ny = y + dy[d];
            bool in_bounds = 0 <= nx && nx < N && 0 <= ny && ny < N;
            if (in_bounds && !mask[nx * N + ny] && st.owner[nx][ny] == player) {{
                mask[nx * N + ny] = 1;
                q.push({{nx, ny}});
            }}
        }}
    }}
    return mask;
}}

array<float, MAX_PLAYERS> player_scores(const State& st) {{
    array<float, MAX_PLAYERS> scores{{}};
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) {{
            int p = st.owner[i][j];
            if (p >= 0) scores[p] += static_cast<float>(st.values[i][j] * st.level[i][j]);
        }}
    }}
    return scores;
}}

array<int, MAX_PLAYERS> player_id_map(const array<float, MAX_PLAYERS>& scores, int m) {{
    vector<int> enemies;
    for (int p = 1; p < m; ++p) enemies.push_back(p);
    sort(enemies.begin(), enemies.end(), [&](int a, int b) {{
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    }});
    array<int, MAX_PLAYERS> mapped;
    mapped.fill(0);
    mapped[0] = 0;
    for (int i = 0; i < static_cast<int>(enemies.size()); ++i) mapped[enemies[i]] = i + 1;
    return mapped;
}}

struct Particle {{
    double wa = 0.0;
    double wb = 0.0;
    double wc = 0.0;
    double wd = 0.0;
    double eps = 0.0;
}};

struct SplitMix64 {{
    uint64_t state;
    bool has_spare = false;
    double spare = 0.0;

    explicit SplitMix64(uint64_t seed) : state(seed) {{}}

    uint64_t next_u64() {{
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }}

    double next_f64() {{
        return static_cast<double>(next_u64() >> 11) * (1.0 / static_cast<double>(1ULL << 53));
    }}

    double uniform(double low, double high) {{
        return low + (high - low) * next_f64();
    }}

    double normal() {{
        if (has_spare) {{
            has_spare = false;
            return spare;
        }}
        double u1 = max(next_f64(), numeric_limits<double>::min());
        double u2 = next_f64();
        double radius = sqrt(-2.0 * log(u1));
        double theta = 2.0 * acos(-1.0) * u2;
        spare = radius * sin(theta);
        has_spare = true;
        return radius * cos(theta);
    }}
}};

double particle_score(const State& st, int player, int x, int y, const Particle& particle) {{
    int owner = st.owner[x][y];
    int level = st.level[x][y];
    double value = static_cast<double>(st.values[x][y]);
    if (owner == -1) return value * particle.wa;
    if (owner == player) return level < st.u ? value * particle.wb : 0.0;
    if (level == 1) return value * particle.wc;
    return value * particle.wd;
}}

void add_policy_distribution(
    const State& st,
    int player,
    const vector<pair<int, int>>& candidates,
    const Particle& particle,
    double weight,
    array<double, N * N>& dist
) {{
    if (candidates.empty()) return;
    double random_prob = particle.eps / static_cast<double>(candidates.size());
    for (auto [x, y] : candidates) dist[x * N + y] += weight * random_prob;

    vector<double> scores;
    scores.reserve(candidates.size());
    double best_score = -numeric_limits<double>::infinity();
    for (auto [x, y] : candidates) {{
        double score = particle_score(st, player, x, y, particle);
        best_score = max(best_score, score);
        scores.push_back(score);
    }}
    double tolerance = 1e-9 * max(abs(best_score), 1.0);
    int best_count = 0;
    for (double score : scores) {{
        if (score >= best_score - tolerance) ++best_count;
    }}
    best_count = max(best_count, 1);
    double greedy_prob = (1.0 - particle.eps) / static_cast<double>(best_count);
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {{
        if (scores[i] >= best_score - tolerance) {{
            auto [x, y] = candidates[i];
            dist[x * N + y] += weight * greedy_prob;
        }}
    }}
}}

struct ParticleFilterSmc {{
    vector<Particle> particles;
    vector<double> weights;
    SplitMix64 rng;

    ParticleFilterSmc(int n, uint64_t seed) : rng(seed) {{
        n = max(n, 1);
        particles.reserve(n);
        for (int i = 0; i < n; ++i) {{
            particles.push_back(Particle{{
                rng.uniform(0.3, 1.0),
                rng.uniform(0.3, 1.0),
                rng.uniform(0.3, 1.0),
                rng.uniform(0.3, 1.0),
                rng.uniform(0.1, 0.5),
            }});
        }}
        weights.assign(n, 1.0 / static_cast<double>(n));
    }}

    Particle mean() const {{
        Particle m;
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {{
            m.wa += weights[i] * particles[i].wa;
            m.wb += weights[i] * particles[i].wb;
            m.wc += weights[i] * particles[i].wc;
            m.wd += weights[i] * particles[i].wd;
            m.eps += weights[i] * particles[i].eps;
        }}
        return m;
    }}

    Particle stddev(const Particle& m) const {{
        Particle v;
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {{
            v.wa += weights[i] * pow(particles[i].wa - m.wa, 2);
            v.wb += weights[i] * pow(particles[i].wb - m.wb, 2);
            v.wc += weights[i] * pow(particles[i].wc - m.wc, 2);
            v.wd += weights[i] * pow(particles[i].wd - m.wd, 2);
            v.eps += weights[i] * pow(particles[i].eps - m.eps, 2);
        }}
        return Particle{{sqrt(max(v.wa, 0.0)), sqrt(max(v.wb, 0.0)), sqrt(max(v.wc, 0.0)),
                        sqrt(max(v.wd, 0.0)), sqrt(max(v.eps, 0.0))}};
    }}

    double ess() const {{
        double sum_sq = 0.0;
        for (double w : weights) sum_sq += w * w;
        return sum_sq <= 0.0 ? 0.0 : 1.0 / sum_sq;
    }}

    void update(const State& st, int player, pair<int, int> observed) {{
        vector<pair<int, int>> candidates = get_candidates(st, player);
        auto it = find(candidates.begin(), candidates.end(), observed);
        if (it == candidates.end() || candidates.empty()) return;
        int obs_idx = static_cast<int>(it - candidates.begin());

        vector<double> logs;
        logs.reserve(particles.size());
        double max_log = -numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {{
            array<double, N * N> dist{{}};
            add_policy_distribution(st, player, candidates, particles[i], 1.0, dist);
            auto [x, y] = candidates[obs_idx];
            double prob = max(dist[x * N + y], 1e-300);
            double log_w = log(max(weights[i], 1e-300)) + log(prob);
            max_log = max(max_log, log_w);
            logs.push_back(log_w);
        }}

        double sum = 0.0;
        for (int i = 0; i < static_cast<int>(weights.size()); ++i) {{
            weights[i] = exp(logs[i] - max_log);
            sum += weights[i];
        }}
        if (!isfinite(sum) || sum <= 0.0) {{
            fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(weights.size()));
            return;
        }}
        for (double& w : weights) w /= sum;
        if (ess() < 0.5 * static_cast<double>(particles.size())) resample();
    }}

    void resample() {{
        int n = static_cast<int>(particles.size());
        Particle m = mean();
        Particle s = stddev(m);
        vector<double> cumulative(n);
        partial_sum(weights.begin(), weights.end(), cumulative.begin());
        cumulative.back() = 1.0;
        double step = 1.0 / static_cast<double>(n);
        double u = rng.next_f64() * step;
        double a = 0.98;
        double h = sqrt(1.0 - a * a);
        int idx = 0;
        vector<Particle> next;
        next.reserve(n);
        auto jitter = [&](double value, double mean_value, double sd, double low, double high) {{
            double center = a * value + (1.0 - a) * mean_value;
            return min(high, max(low, center + h * sd * rng.normal()));
        }};
        for (int i = 0; i < n; ++i) {{
            while (idx + 1 < n && cumulative[idx] < u) ++idx;
            Particle p = particles[idx];
            next.push_back(Particle{{
                jitter(p.wa, m.wa, s.wa, 0.3, 1.0),
                jitter(p.wb, m.wb, s.wb, 0.3, 1.0),
                jitter(p.wc, m.wc, s.wc, 0.3, 1.0),
                jitter(p.wd, m.wd, s.wd, 0.3, 1.0),
                jitter(p.eps, m.eps, s.eps, 0.1, 0.5),
            }});
            u += step;
        }}
        particles = move(next);
        fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(n));
    }}

    array<float, N * N> predictive_distribution(const State& st, int player) const {{
        array<double, N * N> tmp{{}};
        vector<pair<int, int>> candidates = get_candidates(st, player);
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {{
            add_policy_distribution(st, player, candidates, particles[i], weights[i], tmp);
        }}
        array<float, N * N> out{{}};
        for (int i = 0; i < N * N; ++i) out[i] = static_cast<float>(tmp[i]);
        return out;
    }}
}};

array<unsigned char, N * N> reach_mask(const State& st, int player) {{
    array<unsigned char, N * N> mask{{}};
    for (auto [x, y] : get_candidates(st, player)) mask[x * N + y] = 1;
    return mask;
}}

array<float, N * N> dist_to_sources(const array<unsigned char, N * N>& sources) {{
    constexpr int INF = 1 << 20;
    array<int, N * N> dist;
    dist.fill(INF);
    bool has_source = false;
    for (int idx = 0; idx < N * N; ++idx) {{
        if (sources[idx]) {{
            dist[idx] = 0;
            has_source = true;
        }}
    }}
    array<float, N * N> out{{}};
    if (!has_source) {{
        out.fill(1.0f);
        return out;
    }}
    for (int x = 0; x < N; ++x) {{
        for (int y = 0; y < N; ++y) {{
            int idx = x * N + y;
            if (x > 0) dist[idx] = min(dist[idx], dist[(x - 1) * N + y] + 1);
            if (y > 0) dist[idx] = min(dist[idx], dist[x * N + y - 1] + 1);
        }}
    }}
    for (int x = N - 1; x >= 0; --x) {{
        for (int y = N - 1; y >= 0; --y) {{
            int idx = x * N + y;
            if (x + 1 < N) dist[idx] = min(dist[idx], dist[(x + 1) * N + y] + 1);
            if (y + 1 < N) dist[idx] = min(dist[idx], dist[x * N + y + 1] + 1);
        }}
    }}
    for (int idx = 0; idx < N * N; ++idx) out[idx] = dist[idx] >= INF ? 1.0f : dist[idx] / 18.0f;
    return out;
}}

torch::Tensor encode(
    const State& st,
    const vector<pair<int, int>>& candidates,
    const vector<ParticleFilterSmc>& pfilters
) {{
    vector<float> planes(NUM_PLANES * N * N, 0.0f);
    auto at = [&](int plane, int x, int y) -> float& {{
        return planes[(plane * N + x) * N + y];
    }};

    float mean_values = 0.0f;
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) mean_values += st.values[i][j];
    }}
    mean_values /= static_cast<float>(N * N);
    mean_values = max(mean_values, 1.0f);
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) at(0, i, j) = st.values[i][j] / mean_values;
    }}

    auto scores = player_scores(st);
    auto mapped = player_id_map(scores, st.m);

    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) {{
            int owner = st.owner[i][j];
            int mapped_owner = owner < 0 ? -1 : mapped[owner];
            at(mapped_owner + 2, i, j) = 1.0f;
            int lv = st.level[i][j];
            if (1 <= lv && lv <= MAX_LEVEL) at(9 + lv, i, j) = 1.0f;
        }}
    }}

    for (int p = 0; p < st.m; ++p) {{
        int mp = mapped[p];
        auto [x, y] = st.pos[p];
        if (0 <= mp && mp < MAX_PLAYERS && 0 <= x && x < N && 0 <= y && y < N) {{
            at(15 + mp, x, y) = 1.0f;
        }}
    }}

    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) {{
            at(23, i, j) = (100.0f - static_cast<float>(st.turn)) / 100.0f;
            at(PLANE_M, i, j) = static_cast<float>(st.m) / MAX_PLAYERS;
            at(PLANE_U, i, j) = static_cast<float>(st.u) / MAX_LEVEL;
        }}
    }}

    float player0_score = scores[0];
    float max_ai_score = 0.0f;
    for (int p = 1; p < st.m; ++p) max_ai_score = max(max_ai_score, scores[p]);
    float total_capacity = 0.0f;
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) total_capacity += st.values[i][j] * max(st.u, 1);
    }}
    total_capacity = max(total_capacity, 1.0f);
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) {{
            at(PLANE_SCORE_RATIO, i, j) = player0_score / max(max_ai_score, 1.0f);
            at(PLANE_SCORE_DIFF, i, j) = (player0_score - max_ai_score) / total_capacity;
        }}
    }}

    for (auto [x, y] : candidates) at(PLANE_LEGAL_MASK, x, y) = 1.0f;
    for (int p = 0; p < st.m; ++p) {{
        int mp = mapped[p];
        if (0 <= mp && mp < MAX_PLAYERS) {{
            float normalized_score = scores[p] / total_capacity;
            for (int i = 0; i < N; ++i) {{
                for (int j = 0; j < N; ++j) {{
                    at(PLANE_PLAYER_SCORE_START + mp, i, j) = normalized_score;
                }}
            }}
            int param_start = PLANE_ORACLE_PARAM_START + mp * ORACLE_PARAMS_PER_PLAYER;
            Particle params = p > 0 ? pfilters[p - 1].mean() : Particle{{}};
            array<float, ORACLE_PARAMS_PER_PLAYER> values{{
                static_cast<float>(params.wa),
                static_cast<float>(params.wb),
                static_cast<float>(params.wc),
                static_cast<float>(params.wd),
                static_cast<float>(params.eps),
            }};
            for (int k = 0; k < ORACLE_PARAMS_PER_PLAYER; ++k) {{
                for (int i = 0; i < N; ++i) {{
                    for (int j = 0; j < N; ++j) at(param_start + k, i, j) = values[k];
                }}
            }}
        }}
    }}

    vector<array<unsigned char, N * N>> comp_masks;
    vector<array<unsigned char, N * N>> reach_masks;
    vector<array<float, N * N>> next_planes;
    comp_masks.reserve(st.m);
    reach_masks.reserve(st.m);
    next_planes.reserve(st.m);
    for (int p = 0; p < st.m; ++p) {{
        comp_masks.push_back(connected_component_mask(st, p));
        reach_masks.push_back(reach_mask(st, p));
        if (p == 0) {{
            array<float, N * N> own_next{{}};
            for (int idx = 0; idx < N * N; ++idx) {{
                own_next[idx] = reach_masks.back()[idx] ? 1.0f : 0.0f;
            }}
            next_planes.push_back(own_next);
        }} else {{
            next_planes.push_back(pfilters[p - 1].predictive_distribution(st, p));
        }}
    }}

    for (int p = 0; p < st.m; ++p) {{
        int mp = mapped[p];
        if (mp < 0 || mp >= MAX_PLAYERS) continue;
        array<unsigned char, N * N> owner_sources{{}};
        for (int i = 0; i < N; ++i) {{
            for (int j = 0; j < N; ++j) owner_sources[i * N + j] = st.owner[i][j] == p;
        }}
        auto dist_owner = dist_to_sources(owner_sources);
        auto dist_comp = dist_to_sources(comp_masks[p]);
        for (int idx = 0; idx < N * N; ++idx) {{
            int i = idx / N;
            int j = idx % N;
            at(PLANE_COMP_START + mp, i, j) = comp_masks[p][idx] ? 1.0f : 0.0f;
            at(PLANE_REACH_START + mp, i, j) = reach_masks[p][idx] ? 1.0f : 0.0f;
            at(PLANE_NEXT_GREEDY_START + mp, i, j) = next_planes[p][idx];
            at(PLANE_DIST_OWNER_START + mp, i, j) = dist_owner[idx];
            at(PLANE_DIST_COMP_START + mp, i, j) = dist_comp[idx];
        }}
    }}

    const float inv_board_span = 1.0f / static_cast<float>(N - 1);
    const float pos0_x_norm = static_cast<float>(st.pos[0].first) * inv_board_span;
    const float pos0_y_norm = static_cast<float>(st.pos[0].second) * inv_board_span;
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) {{
            float dx = abs(static_cast<float>(i) - 4.5f);
            float dy = abs(static_cast<float>(j) - 4.5f);
            at(PLANE_DIST_CENTER, i, j) = (dx + dy) / 9.0f;
            at(PLANE_X_NORM, i, j) = static_cast<float>(i) * inv_board_span;
            at(PLANE_Y_NORM, i, j) = static_cast<float>(j) * inv_board_span;
            at(PLANE_POS0_X_NORM, i, j) = pos0_x_norm;
            at(PLANE_POS0_Y_NORM, i, j) = pos0_y_norm;
        }}
    }}

    const float level_capacity = max(static_cast<float>(N * N * max(st.u, 1)), 1.0f);
    for (int p = 0; p < st.m; ++p) {{
        int mp = mapped[p];
        if (mp < 0 || mp >= MAX_PLAYERS) continue;
        float owner_level_sum = 0.0f;
        float owner_level_value_sum = 0.0f;
        float comp_level_sum = 0.0f;
        float comp_level_value_sum = 0.0f;
        for (int i = 0; i < N; ++i) {{
            for (int j = 0; j < N; ++j) {{
                float level = static_cast<float>(st.level[i][j]);
                float level_value = level * static_cast<float>(st.values[i][j]);
                if (st.owner[i][j] == p) {{
                    owner_level_sum += level;
                    owner_level_value_sum += level_value;
                }}
                if (comp_masks[p][i * N + j]) {{
                    comp_level_sum += level;
                    comp_level_value_sum += level_value;
                }}
            }}
        }}
        int agg_start = PLANE_PLAYER_AGG_START + mp * PLAYER_AGG_FEATURES;
        for (int i = 0; i < N; ++i) {{
            for (int j = 0; j < N; ++j) {{
                at(agg_start + PLAYER_AGG_OWNER_LEVEL_SUM, i, j) = owner_level_sum / level_capacity;
                at(agg_start + PLAYER_AGG_OWNER_LEVEL_VALUE_SUM, i, j) =
                    owner_level_value_sum / total_capacity;
                at(agg_start + PLAYER_AGG_COMP_LEVEL_SUM, i, j) = comp_level_sum / level_capacity;
                at(agg_start + PLAYER_AGG_COMP_LEVEL_VALUE_SUM, i, j) =
                    comp_level_value_sum / total_capacity;
            }}
        }}
    }}

    return torch::from_blob(planes.data(), {{1, NUM_PLANES, N, N}}, torch::kFloat32)
        .to(torch::kBFloat16);
}}

torch::jit::script::Module load_model() {{
    vector<unsigned char> model_bytes = decode_base91(kEncodedModel);
    string model_data(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());
    istringstream input(model_data, ios::binary);
    torch::NoGradGuard no_grad;
    auto module = torch::jit::load(input, torch::kCPU);
    module.eval();
    return module;
}}

pair<int, int> choose_action(
    torch::jit::script::Module& module,
    const State& st,
    const vector<ParticleFilterSmc>& pfilters,
    mt19937& rng
) {{
    vector<pair<int, int>> candidates = get_candidates(st, 0);
    if (candidates.empty()) return st.pos[0];
    torch::NoGradGuard no_grad;
    torch::Tensor input = encode(st, candidates, pfilters);
    auto output = module.forward({{input}}).toTuple();
    torch::Tensor logits = output->elements()[0].toTensor()
        .to(torch::kFloat32)
        .reshape({{N * N}})
        .contiguous();
    auto acc = logits.accessor<float, 1>();

    if (ACTION_TEMPERATURE <= 0.0) {{
        pair<int, int> best = candidates[0];
        float best_logit = -numeric_limits<float>::infinity();
        for (auto [x, y] : candidates) {{
            int idx = x * N + y;
            if (acc[idx] > best_logit) {{
                best_logit = acc[idx];
                best = {{x, y}};
            }}
        }}
        return best;
    }}

    float max_logit = -numeric_limits<float>::infinity();
    for (auto [x, y] : candidates) {{
        int idx = x * N + y;
        max_logit = max(max_logit, acc[idx]);
    }}
    vector<double> weights;
    weights.reserve(candidates.size());
    for (auto [x, y] : candidates) {{
        int idx = x * N + y;
        weights.push_back(exp(static_cast<double>(acc[idx] - max_logit) / ACTION_TEMPERATURE));
    }}
    discrete_distribution<int> dist(weights.begin(), weights.end());
    return candidates[dist(rng)];
}}

}}  // namespace

int main() {{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    State st;
    int input_n = 0;
    cin >> input_n >> st.m >> st.turn >> st.u;
    st.turn = 0;
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) cin >> st.values[i][j];
    }}
    for (int p = 0; p < st.m; ++p) {{
        cin >> st.pos[p].first >> st.pos[p].second;
    }}
    for (int i = 0; i < N; ++i) {{
        for (int j = 0; j < N; ++j) {{
            st.owner[i][j] = -1;
            st.level[i][j] = 0;
        }}
    }}
    for (int p = 0; p < st.m; ++p) {{
        auto [x, y] = st.pos[p];
        st.owner[x][y] = p;
        st.level[x][y] = 1;
    }}

    auto module = load_model();
    vector<ParticleFilterSmc> pfilters;
    pfilters.reserve(max(st.m - 1, 0));
    for (int p = 1; p < st.m; ++p) {{
        pfilters.emplace_back(
            PF_PARTICLES,
            0xa0761d6478bd642fULL ^ (static_cast<uint64_t>(p) << 32)
        );
    }}
    mt19937 rng(static_cast<uint32_t>(
        chrono::steady_clock::now().time_since_epoch().count()
    ));

    for (int t = 0; t < T; ++t) {{
        st.turn = t;
        auto [x, y] = choose_action(module, st, pfilters, rng);
        cout << x << ' ' << y << endl;

        vector<pair<int, int>> selected(st.m);
        for (int p = 0; p < st.m; ++p) {{
            int tx, ty;
            cin >> tx >> ty;
            selected[p] = {{tx, ty}};
        }}
        for (int p = 1; p < st.m; ++p) {{
            pfilters[p - 1].update(st, p, selected[p]);
        }}
        for (int p = 0; p < st.m; ++p) {{
            cin >> st.pos[p].first >> st.pos[p].second;
        }}
        for (int i = 0; i < N; ++i) {{
            for (int j = 0; j < N; ++j) cin >> st.owner[i][j];
        }}
        for (int i = 0; i < N; ++i) {{
            for (int j = 0; j < N; ++j) cin >> st.level[i][j];
        }}
    }}
    return 0;
}}
"""


def render_q4_cpp(
    encoded_model: str,
    *,
    checkpoint_name: str,
    packed_size: int,
    pf_particles: int,
    temperature: float,
) -> str:
    """Render an actor-only, direct LibTorch q4 inference submission."""
    source = render_cpp(
        encoded_model,
        checkpoint_name=checkpoint_name,
        torchscript_size=packed_size,
        pf_particles=pf_particles,
        temperature=temperature,
    )
    direct_model = r"""
using torch::Tensor;

Tensor l2norm(const Tensor& x, int64_t dim) {
    return x / torch::sqrt(torch::sum(x * x, {dim}, true) + 1e-8);
}

Tensor qlinear(const Tensor& x, const Tensor& weight, const Tensor& bias = Tensor()) {
    Tensor y = torch::matmul(x, weight.t());
    return bias.defined() ? y + bias : y;
}

struct Q4Reader {
    const vector<unsigned char>& data;
    size_t pos = 0;
    explicit Q4Reader(const vector<unsigned char>& source) : data(source) {
        const char magic[] = "AHC061Q4";
        for (int i = 0; i < 8; ++i) {
            if (data.at(pos++) != magic[i]) throw runtime_error("bad q4 model");
        }
        if (data.at(pos++) != 1) throw runtime_error("unsupported q4 model");
        pos += 2;  // tensor count; the fixed actor layout below validates consumption.
    }
    uint32_t u32() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value |= uint32_t(data.at(pos++)) << (8 * i);
        return value;
    }
    float half() {
        uint16_t bits = uint16_t(data.at(pos)) | (uint16_t(data.at(pos + 1)) << 8);
        pos += 2;
        c10::Half value;
        memcpy(&value, &bits, sizeof(bits));
        return static_cast<float>(value);
    }
    Tensor take(vector<int64_t> shape) {
        const int mode = data.at(pos++);
        const uint32_t n = u32();
        int64_t expected = 1;
        for (int64_t d : shape) expected *= d;
        if (n != expected) {
            throw runtime_error(
                "q4 tensor shape mismatch: got " + to_string(n) + " expected " + to_string(expected)
            );
        }
        vector<float> values(n);
        if (mode == 1) {
            for (uint32_t i = 0; i < n; ++i) values[i] = half();
        } else if (mode == 0) {
            const size_t codes = (n + 1) / 2;
            const size_t scales = (n + 127) / 128;
            const size_t code_start = pos;
            pos += codes;
            vector<float> group_scales(scales);
            for (size_t i = 0; i < scales; ++i) group_scales[i] = half();
            for (uint32_t i = 0; i < n; ++i) {
                int q = (data[code_start + i / 2] >> (4 * (i & 1))) & 15;
                if (q >= 8) q -= 16;
                values[i] = q * group_scales[i / 128];
            }
        } else throw runtime_error("bad q4 tensor mode");
        return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
    }
};

struct AttentionBlock {
    Tensor w1, s1, w2, alpha, rel, qkv, out, out_scale, attn_alpha;
    Tensor forward(const Tensor& input) const {
        Tensor x = input;
        Tensor y = x.permute({0, 2, 3, 1});
        y = qlinear(y, w1) * s1;
        y = torch::relu(y) + 1e-8;
        y = l2norm(qlinear(y, w2), -1).permute({0, 3, 1, 2});
        Tensor mixed = x + ((y - x).permute({0, 2, 3, 1}) * alpha).permute({0, 3, 1, 2});
        x = l2norm(mixed, 1);

        y = x.permute({0, 2, 3, 1}).reshape({1, 100, 64});
        Tensor qkv_value = qlinear(y, qkv).view({1, 100, 3, 4, 16});
        Tensor q = l2norm(qkv_value.select(2, 0), -1).permute({0, 2, 1, 3});
        Tensor k = l2norm(qkv_value.select(2, 1), -1).permute({0, 2, 1, 3});
        Tensor v = qkv_value.select(2, 2).permute({0, 2, 1, 3});
        Tensor logits = torch::matmul(q, k.transpose(-2, -1)) * 4.0 + rel;
        Tensor attended = torch::matmul(torch::softmax(logits, -1), v)
            .permute({0, 2, 1, 3}).reshape({1, 100, 64});
        Tensor target = l2norm(qlinear(attended, out) * out_scale, -1)
            .view({1, 10, 10, 64}).permute({0, 3, 1, 2});
        mixed = x + ((target - x).permute({0, 2, 3, 1}) * attn_alpha).permute({0, 3, 1, 2});
        return l2norm(mixed, 1);
    }
};

struct Q4Policy {
    Tensor embed, embed_scale, policy1, gn_weight, gn_bias, policy2, policy2_bias, mean, invstd;
    vector<AttentionBlock> blocks;
    explicit Q4Policy(const vector<unsigned char>& bytes) {
        Q4Reader reader(bytes);
        embed = reader.take({64, 155}); embed_scale = reader.take({64});
        for (int i = 0; i < 8; ++i) {
            AttentionBlock b;
            b.w1 = reader.take({256, 64}); b.s1 = reader.take({256}); b.w2 = reader.take({64, 256});
            b.alpha = reader.take({64}); b.rel = reader.take({4, 19, 19});
            b.qkv = reader.take({192, 64});
            b.out = reader.take({64, 64}); b.out_scale = reader.take({64});
            b.attn_alpha = reader.take({64});
            vector<float> bias(4 * 100 * 100);
            auto a = b.rel.accessor<float, 3>();
            for (int h = 0; h < 4; ++h) for (int i = 0; i < 100; ++i) for (int j = 0; j < 100; ++j)
                bias[(h * 100 + i) * 100 + j] = a[h][i / 10 - j / 10 + 9][i % 10 - j % 10 + 9];
            b.rel = torch::from_blob(bias.data(), {1, 4, 100, 100}, torch::kFloat32).clone();
            blocks.push_back(move(b));
        }
        policy1 = reader.take({64, 64, 1, 1}); gn_weight = reader.take({64});
        gn_bias = reader.take({64});
        policy2 = reader.take({1, 64, 1, 1}); policy2_bias = reader.take({1});
        mean = reader.take({NUM_PLANES, 1, 1}); invstd = reader.take({NUM_PLANES, 1, 1});
    }
    Tensor forward(const Tensor& input) const {
        Tensor raw = input.to(torch::kFloat32);
        Tensor x = (raw - mean) * invstd;
        x = x.permute({0, 2, 3, 1});
        Tensor shift = torch::full({1, 10, 10, 1}, 3.0, torch::kFloat32);
        x = l2norm(torch::cat({x, shift}, -1), -1);
        x = l2norm(qlinear(x, embed) * embed_scale, -1).permute({0, 3, 1, 2});
        for (const auto& block : blocks) x = block.forward(x);
        x = qlinear(x.permute({0, 2, 3, 1}), policy1.reshape({64, 64})).permute({0, 3, 1, 2});
        Tensor grouped = x.view({1, 8, 8, 10, 10});
        Tensor mu = grouped.mean({2, 3, 4}, true);
        Tensor centered = grouped - mu;
        x = (centered / torch::sqrt((centered * centered).mean({2, 3, 4}, true) + 1e-5))
            .view({1, 64, 10, 10});
        x = x * gn_weight.view({1, 64, 1, 1}) + gn_bias.view({1, 64, 1, 1});
        x = torch::relu(x).permute({0, 2, 3, 1});
        return qlinear(x, policy2.reshape({1, 64}), policy2_bias).reshape({1, 100});
    }
};
"""
    source = source.replace("\nstruct State {", "\n" + direct_model + "\nstruct State {")
    source = source.replace(
        "torch::jit::script::Module load_model() {\n"
        "    vector<unsigned char> model_bytes = decode_base91(kEncodedModel);\n"
        "    string model_data(\n"
        "        reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());\n"
        "    istringstream input(model_data, ios::binary);\n"
        "    torch::NoGradGuard no_grad;\n"
        "    auto module = torch::jit::load(input, torch::kCPU);\n"
        "    module.eval();\n"
        "    return module;\n"
        "}",
        "Q4Policy load_model() { return Q4Policy(decode_base91(kEncodedModel)); }",
    )
    source = source.replace("torch::jit::script::Module& module,", "Q4Policy& module,")
    source = source.replace(
        "auto output = module.forward({input}).toTuple();\n"
        "    torch::Tensor logits = output->elements()[0].toTensor()\n"
        "        .to(torch::kFloat32)\n"
        "        .reshape({N * N})\n"
        "        .contiguous();",
        "torch::Tensor logits = module.forward(input).reshape({N * N}).contiguous();",
    )
    source = source.replace(
        "torch::jit::script::Module module = load_model();", "Q4Policy module = load_model();"
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pf-particles", type=int)
    parser.add_argument(
        "--quantized-q4",
        action="store_true",
        help="emit an actor-only 4-bit grouped-quantized submission",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax sampling temperature. Use <= 0 for greedy argmax.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint = args.checkpoint or (run_dir / "checkpoint_latest.pt")
    output = args.output or (run_dir / "submit.cpp")

    if args.quantized_q4:
        packed = pack_q4_policy(checkpoint, config)
        encoded = base91_encode(packed)
        rendered = render_q4_cpp(
            encoded,
            checkpoint_name=checkpoint.stem.replace("step_", "s").replace(
                "checkpoint_latest",
                "latest",
            ),
            packed_size=len(packed),
            pf_particles=int(args.pf_particles or config.get("pf_particles", 16)),
            temperature=args.temperature,
        )
        print(f"q4_packed_bytes={len(packed)}")
    else:
        torchscript = export_torchscript(checkpoint, config)
        encoded = base91_encode(torchscript)
        rendered = render_cpp(
            encoded,
            checkpoint_name=checkpoint.stem.replace("step_", "s").replace(
                "checkpoint_latest",
                "latest",
            ),
            torchscript_size=len(torchscript),
            pf_particles=int(args.pf_particles or config.get("pf_particles", 16)),
            temperature=args.temperature,
        )
        print(f"torchscript_bytes={len(torchscript)}")
    output.write_text(rendered)
    print(f"checkpoint={checkpoint}")
    print(f"base91_chars={len(encoded)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
