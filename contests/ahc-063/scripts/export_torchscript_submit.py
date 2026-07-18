"""Export an AHC063 PPO checkpoint as a self-contained libtorch submit.cpp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ahcrl.contests.ahc063.encoder import NUM_PLANES
from ahcrl.contests.ahc063.model import ActorCritic, RunningObservationNormalizer

ROOT = Path(__file__).resolve().parents[3]
MAX_BOARD_SIZE = 16
MAX_COLORS = 7
ACTION_COUNT = 4
BASE91_ALPHABET = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~"'
)


class NormalizedPolicy(nn.Module):
    def __init__(self, model: ActorCritic, normalizer: RunningObservationNormalizer | None) -> None:
        super().__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalizer is not None:
            mean = self.normalizer.mean.to(device=x.device, dtype=torch.float32)
            count = self.normalizer.count.to(dtype=self.normalizer.m2.dtype).clamp_min(1)
            variance = torch.where(
                self.normalizer.count > 0,
                self.normalizer.m2 / count,
                torch.ones_like(self.normalizer.m2),
            ).to(device=x.device)
            x = (x.float() - mean) / torch.sqrt(variance + self.normalizer.epsilon)
        return self.model(x)


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
            out.extend((BASE91_ALPHABET[value % 91], BASE91_ALPHABET[value // 91]))
    if bit_count:
        out.append(BASE91_ALPHABET[bit_queue % 91])
        if bit_count > 7 or bit_queue > 90:
            out.append(BASE91_ALPHABET[bit_queue // 91])
    return "".join(out)


def c_string_chunks(value: str, width: int = 120) -> str:
    chunks = []
    for start in range(0, len(value), width):
        chunk = value[start : start + width].replace("\\", "\\\\").replace('"', '\\"')
        chunks.append(f'    "{chunk}"')
    return "\n".join(chunks)


def load_policy(checkpoint: Path, config: dict[str, Any]) -> NormalizedPolicy:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(
        channels=int(config["model_channels"]),
        blocks=int(config["model_blocks"]),
        block_type=str(config.get("model_block_type", "convnext")),
    ).float()
    normalizer: RunningObservationNormalizer | None = None
    if bool(config.get("obs_norm", False)):
        normalizer = RunningObservationNormalizer(
            NUM_PLANES,
            epsilon=float(config.get("obs_norm_epsilon", 1e-8)),
        ).float()
    model.observation_normalizer = normalizer
    model.load_state_dict(state["model"])
    model.eval()
    policy = NormalizedPolicy(model, normalizer).eval().float()
    return policy


def export_torchscript(checkpoint: Path, config: dict[str, Any]) -> bytes:
    policy = load_policy(checkpoint, config)
    dummy = torch.zeros((1, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(policy, dummy, strict=True)
        frozen = torch.jit.freeze(traced)
    output = Path("/tmp/ahc063_export.pt")
    frozen.save(str(output))
    return output.read_bytes()


CPP_TEMPLATE = r"""#include <ATen/Parallel.h>
#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {
constexpr int MAX_N = 16;
constexpr int MAX_COLORS = 7;
constexpr int NUM_PLANES = 43;
constexpr int MAX_STEPS = 100000;
const string kAlphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\"";
const string kModel =
@MODEL@
;

vector<unsigned char> decode_base91(const string& input) {
    array<int, 256> decode;
    decode.fill(-1);
    for (int i = 0; i < 91; ++i) decode[static_cast<unsigned char>(kAlphabet[i])] = i;
    vector<unsigned char> output;
    int value = -1;
    unsigned int queue = 0;
    int bits = 0;
    for (unsigned char ch : input) {
        const int decoded = decode[ch];
        if (decoded < 0) continue;
        if (value < 0) {
            value = decoded;
        } else {
            value += decoded * 91;
            queue |= static_cast<unsigned int>(value) << bits;
            bits += (value & 8191) > 88 ? 13 : 14;
            while (bits >= 8) {
                output.push_back(static_cast<unsigned char>(queue & 255));
                queue >>= 8;
                bits -= 8;
            }
            value = -1;
        }
    }
    if (value >= 0) output.push_back(static_cast<unsigned char>((queue | (value << bits)) & 255));
    return output;
}

struct Snake {
    int n = 0;
    int m = 0;
    int c = 0;
    int steps = 0;
    int previous_action = -1;
    vector<int> desired;
    int food[MAX_N][MAX_N]{};
    vector<pair<int, int>> position;
    vector<int> color;
};

torch::jit::script::Module load_model() {
    auto bytes = decode_base91(kModel);
    string data(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    istringstream input(data);
    return torch::jit::load(input, torch::kCPU);
}

int food_count(const Snake& snake) {
    int count = 0;
    for (int i = 0; i < snake.n; ++i)
        for (int j = 0; j < snake.n; ++j)
            count += snake.food[i][j] != 0;
    return count;
}

bool target_sequence_matches(const Snake& snake) {
    if (snake.position.size() != static_cast<size_t>(snake.m)) return false;
    for (int index = 0; index < snake.m; ++index)
        if (snake.color[index] != snake.desired[index]) return false;
    return true;
}

array<bool, 4> legal_actions(const Snake& snake) {
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    array<bool, 4> legal{};
    const auto [row, col] = snake.position[0];
    for (int action = 0; action < 4; ++action) {
        const int nr = row + dr[action];
        const int nc = col + dc[action];
        legal[action] = 0 <= nr && nr < snake.n && 0 <= nc && nc < snake.n;
        if (legal[action] && snake.position.size() > 1 &&
            snake.position[1] == make_pair(nr, nc)) legal[action] = false;
    }
    return legal;
}

void encode(const Snake& snake, vector<float>& planes) {
    fill(planes.begin(), planes.end(), 0.0f);
    auto at = [&](int plane, int row, int col) -> float& {
        return planes[(plane * MAX_N + row) * MAX_N + col];
    };
    for (int row = 0; row < snake.n; ++row) {
        for (int col = 0; col < snake.n; ++col) {
            const int food = snake.food[row][col];
            if (food > 0) at(food - 1, row, col) = 1.0f;
        }
    }
    const int length = static_cast<int>(snake.position.size());
    for (int index = 0; index < length; ++index) {
        const auto [row, col] = snake.position[index];
        const int color = snake.color[index];
        at(7 + color - 1, row, col) = 1.0f;
        at(14, row, col) = index == 0;
        at(15, row, col) = 0 < index && index < length - 1;
        at(16, row, col) = index == length - 1;
    }
    if (length < snake.m) {
        const int target = snake.desired[length];
        at(16 + target, 0, 0) = 1.0f;
        for (int row = 0; row < snake.n; ++row)
            for (int col = 0; col < snake.n; ++col)
                at(16 + target, row, col) = 1.0f;
    }
    for (int color = 1; color <= snake.c; ++color) {
        int remaining = 0;
        for (int index = length; index < snake.m; ++index)
            remaining += snake.desired[index] == color;
        for (int row = 0; row < snake.n; ++row)
            for (int col = 0; col < snake.n; ++col)
                at(23 + color, row, col) = static_cast<float>(remaining) / max(snake.m, 1);
    }
    const auto [head_row, head_col] = snake.position[0];
    const float scalars[8] = {
        static_cast<float>(snake.n) / MAX_N,
        static_cast<float>(snake.c) / MAX_COLORS,
        static_cast<float>(length) / max(snake.m, 1),
        static_cast<float>(min(length, snake.m)) / max(snake.m, 1),
        static_cast<float>(head_row) / max(snake.n - 1, 1),
        static_cast<float>(head_col) / max(snake.n - 1, 1),
        static_cast<float>(food_count(snake)) / max(snake.m - 5, 1),
        static_cast<float>(snake.steps) / MAX_STEPS,
    };
    for (int index = 0; index < 8; ++index)
        for (int row = 0; row < snake.n; ++row)
            for (int col = 0; col < snake.n; ++col)
                at(31 + index, row, col) = scalars[index];
    if (snake.previous_action >= 0)
        for (int row = 0; row < snake.n; ++row)
            for (int col = 0; col < snake.n; ++col)
                at(39 + snake.previous_action, row, col) = 1.0f;
}

void step(Snake& snake, int action) {
    static constexpr int dr[4] = {-1, 1, 0, 0};
    static constexpr int dc[4] = {0, 0, -1, 1};
    ++snake.steps;
    snake.previous_action = action;
    const int old_length = static_cast<int>(snake.position.size());
    const auto old_position = snake.position;
    const auto old_color = snake.color;
    vector<pair<int, int>> moved(old_length);
    moved[0] = {old_position[0].first + dr[action], old_position[0].second + dc[action]};
    for (int index = 1; index < old_length; ++index) moved[index] = old_position[index - 1];
    int collision = -1;
    for (int index = 1; index < old_length; ++index)
        if (moved[index] == moved[0]) { collision = index; break; }
    if (collision >= 0) {
        for (int index = collision + 1; index < old_length; ++index)
            snake.food[moved[index].first][moved[index].second] = old_color[index];
        snake.position.assign(moved.begin(), moved.begin() + collision + 1);
        snake.color.assign(old_color.begin(), old_color.begin() + collision + 1);
        return;
    }
    const int food = snake.food[moved[0].first][moved[0].second];
    if (food != 0) {
        snake.food[moved[0].first][moved[0].second] = 0;
        snake.position = moved;
        snake.position.push_back(old_position.back());
        snake.color = old_color;
        snake.color.push_back(food);
    } else {
        snake.position = moved;
        snake.color = old_color;
    }
}
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    Snake snake;
    if (!(cin >> snake.n >> snake.m >> snake.c)) return 0;
    snake.desired.resize(snake.m);
    for (int& value : snake.desired) cin >> value;
    for (int row = 0; row < snake.n; ++row)
        for (int col = 0; col < snake.n; ++col) cin >> snake.food[row][col];
    snake.position.resize(5);
    snake.color.assign(5, 1);
    for (int index = 0; index < 5; ++index) snake.position[index] = {4 - index, 0};

    auto module = load_model();
    torch::NoGradGuard no_grad;
    vector<float> planes(NUM_PLANES * MAX_N * MAX_N);
    while (snake.steps < MAX_STEPS &&
           (food_count(snake) > 0 || !target_sequence_matches(snake))) {
        encode(snake, planes);
        auto input = torch::from_blob(
            planes.data(), {1, NUM_PLANES, MAX_N, MAX_N},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();
        auto output = module.forward({input}).toTuple();
        auto logits = output->elements()[0].toTensor().contiguous();
        auto legal = legal_actions(snake);
        int action = -1;
        float best = -numeric_limits<float>::infinity();
        for (int candidate = 0; candidate < 4; ++candidate) {
            if (legal[candidate] && logits[0][candidate].item<float>() > best) {
                best = logits[0][candidate].item<float>();
                action = candidate;
            }
        }
        if (action < 0) break;
        static constexpr char directions[4] = {'U', 'D', 'L', 'R'};
        cout << directions[action] << '\n';
        step(snake, action);
    }
    return 0;
}
"""


def find_latest_run(artifact_dir: Path) -> Path:
    runs = [path for path in artifact_dir.glob("run_*") if path.is_dir()]
    runs = [path for path in runs if (path / "checkpoints" / "checkpoint_latest.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"no completed PPO run found under {artifact_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or find_latest_run(ROOT / "contests/ahc-063/artifacts/ppo")
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint = run_dir / "checkpoints" / "checkpoint_latest.pt"
    output = args.output or run_dir / "submit.cpp"
    model_bytes = export_torchscript(checkpoint, config)
    output.write_text(CPP_TEMPLATE.replace("@MODEL@", c_string_chunks(base91_encode(model_bytes))))
    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint}")
    print(f"torchscript_bytes={len(model_bytes)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
