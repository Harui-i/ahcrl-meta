from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from ahcrl.contests.ahc061.encoder import NUM_PLANES
from ahcrl.contests.ahc061.model import ActorCritic

BASE91_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
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


def export_torchscript(checkpoint_path: Path, config: dict[str, object]) -> bytes:
    model = ActorCritic(
        channels=int(config["model_channels"]),
        blocks=int(config["model_blocks"]),
        block_type=str(config["model_block_type"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(dtype=torch.bfloat16)
    model.eval()
    dummy = torch.zeros((1, NUM_PLANES, 10, 10), dtype=torch.bfloat16)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=True)
        frozen = torch.jit.freeze(traced)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        frozen.save(f.name)
        return Path(f.name).read_bytes()


def render_cpp(encoded_model: str, *, checkpoint_name: str, torchscript_size: int) -> str:
    encoded_chunks = c_string_literal_chunks(encoded_model)
    return f"""#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <queue>
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
constexpr int ORACLE_PARAMS_PER_PLAYER = 5;
constexpr int PLANE_M = 24;
constexpr int PLANE_U = 25;
constexpr int PLANE_SCORE_RATIO = 26;
constexpr int PLANE_SCORE_DIFF = 27;
constexpr int PLANE_LEGAL_MASK = 28;
constexpr int PLANE_PLAYER_SCORE_START = 29;
constexpr int PLANE_ORACLE_PARAM_START = PLANE_PLAYER_SCORE_START + MAX_PLAYERS;

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

torch::Tensor encode(const State& st, const vector<pair<int, int>>& candidates) {{
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
        }}
    }}
    // 本番入力ではAI内部パラメータは観測できないため、oracle parameter planesは0のままにする。

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

pair<int, int> choose_action(torch::jit::script::Module& module, const State& st) {{
    vector<pair<int, int>> candidates = get_candidates(st, 0);
    if (candidates.empty()) return st.pos[0];
    torch::NoGradGuard no_grad;
    torch::Tensor input = encode(st, candidates);
    auto output = module.forward({{input}}).toTuple();
    torch::Tensor logits = output->elements()[0].toTensor()
        .to(torch::kFloat32)
        .reshape({{N * N}})
        .contiguous();
    auto acc = logits.accessor<float, 1>();

    int best = candidates[0].first * N + candidates[0].second;
    float best_score = acc[best];
    for (auto [x, y] : candidates) {{
        int idx = x * N + y;
        float score = acc[idx];
        if (score > best_score) {{
            best_score = score;
            best = idx;
        }}
    }}
    return {{best / N, best % N}};
}}

}}  // namespace

int main() {{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

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

    for (int t = 0; t < T; ++t) {{
        st.turn = t;
        auto [x, y] = choose_action(module, st);
        cout << x << ' ' << y << endl;

        for (int p = 0; p < st.m; ++p) {{
            int tx, ty;
            cin >> tx >> ty;
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint = args.checkpoint or (run_dir / "checkpoint_latest.pt")
    output = args.output or (run_dir / "submit.cpp")

    torchscript = export_torchscript(checkpoint, config)
    encoded = base91_encode(torchscript)
    output.write_text(
        render_cpp(
            encoded,
            checkpoint_name=checkpoint.stem.replace("step_", "s").replace(
                "checkpoint_latest",
                "latest",
            ),
            torchscript_size=len(torchscript),
        )
    )
    print(f"checkpoint={checkpoint}")
    print(f"torchscript_bytes={len(torchscript)}")
    print(f"base91_chars={len(encoded)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
