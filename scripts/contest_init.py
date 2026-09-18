"""Initialize the common layout for a new AtCoder Heuristic Contest."""

# The generated TOML and README templates intentionally contain long lines.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def normalize_contest(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"(?:ahc[-_]?)?(\d{1,3})", value.strip().lower())
    if match is None:
        raise ValueError(f"invalid contest name: {value!r}; use e.g. ahc068")

    number = int(match.group(1))
    if not 0 <= number <= 999:
        raise ValueError(f"contest number must be between 0 and 999: {number}")

    number_text = f"{number:03d}"
    return f"ahc{number_text}", f"ahc-{number_text}"


def render(template: str, *, slug: str, directory: str) -> str:
    factory_name = f"Ahc{slug.removeprefix('ahc')}Factory"
    env_name = f"Ahc{slug.removeprefix('ahc')}Env"
    return (
        template.replace("__SLUG__", slug)
        .replace("__DIRECTORY__", directory)
        .replace("__FACTORY__", factory_name)
        .replace("__ENV__", env_name)
    )


def write_if_missing(path: Path, content: str, created: list[Path]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    created.append(path)


def download_tools(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ahcrl-meta contest-init"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()

    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        for member in zip_file.infolist():
            relative = PurePosixPath(member.filename)
            if not relative.parts or relative.parts[0] != "tools":
                continue

            output = destination.joinpath(*relative.parts)
            try:
                output.resolve().relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"unsafe path in tools archive: {member.filename}") from error

            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if output.exists():
                continue

            output.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)


def starter_files(slug: str, directory: str) -> dict[str, str]:
    files = {
        f"contests/{directory}/AGENTS.md": f"""# {slug.upper()}\n\n問題文とコンテスト固有の制約を確認してから実装する。\n""",
        f"contests/{directory}/README.md": f"""# {slug.upper()}\n\n新しいAHCの作業ディレクトリ。\n\n- コンテスト: https://atcoder.jp/contests/{slug}\n- 問題文: [problem_ja.md](problem_ja.md) / [problem_en.md](problem_en.md)\n- 公式tools: `tools/`（取得後は原則編集しない）\n- Rust RL環境: `rl-tools/`（`ahcrl-env-core` protocolを実装する）\n- PPO設定: `configs/`（`ahcrl.training` の名前空間付きTOML）\n- Pahcer設定: `eval/pahcer_config.toml`\n- PPO成果物: `artifacts/ppo/`（git管理外）\n\n## 最初にすること\n\n1. `problem_ja.md` と `problem_en.md` に問題文を保存する。\n2. `tools/` にAtCoder公式toolsを配置し、`cargo build --release --manifest-path tools/Cargo.toml`する。\n3. `rl-tools/src/lib.rs` の `EnvFactory` / `ContestEnv` を、公式toolsを使う simulator・観測・action space・metricsで実装する。\n4. `src/ahcrl/contests/{slug}/` のencoder・model・`train_ppo.py`を実装し、`ahcrl.envs.RustVecEnv` と `ahcrl.training` を利用する。\n5. Rust環境の実装後に `make contest-check CONTEST={directory}` を通す。\n6. `eval/main.cpp`を公式visで検証し、`make check`を通してからPPOを開始する。\n""",
        f"contests/{directory}/problem_ja.md": "# 問題文（日本語）\n\nTODO: AtCoderの問題文を保存する。\n",
        f"contests/{directory}/problem_en.md": "# Problem Statement (English)\n\nTODO: Save the AtCoder problem statement here.\n",
        f"contests/{directory}/configs/ppo_smoke.toml": render(
            """[training]\nnum_envs = 4\ntotal_steps = 4096\nrollout_steps = 32\nseed_start = 0\nseed_stride = 1\ndevice = \"cpu\"\ncompile = false\nartifact_dir = \"contests/__DIRECTORY__/artifacts/ppo\"\ncheckpoint_interval_updates = 1\n\n[ppo]\nlr = 0.0003\ngamma = 0.99\ngae_lambda = 0.95\nclip = 0.2\nepochs = 1\nminibatch_size = 32\nentropy_coef = 0.01\nvalue_coef = 0.5\nmax_grad_norm = 0.5\nproximal_ewma = true\nproximal_ewma_com = 256.0\n\n[wandb]\nenabled = false\nproject = \"ahcrl-meta-__SLUG__\"\nname = \"__SLUG__-ppo-ewma-smoke-com256\"\ntags = [\"__SLUG__\", \"ppo-ewma\", \"ewma\", \"smoke\"]\n\n[model]\nchannels = 32\nblocks = 2\n""",
            slug=slug,
            directory=directory,
        ),
        f"contests/{directory}/configs/ppo_train.toml": render(
            """[training]\nnum_envs = 256\ntotal_steps = 20000000\nrollout_steps = 128\nseed_start = 0\nseed_stride = 1\ndevice = \"cuda\"\ncompile = true\nartifact_dir = \"contests/__DIRECTORY__/artifacts/ppo\"\ncheckpoint_interval_updates = 40\n\n[ppo]\nlr = 0.0003\ngamma = 0.99\ngae_lambda = 0.95\nclip = 0.2\nepochs = 1\nminibatch_size = 1024\nentropy_coef = 0.001\nvalue_coef = 0.5\nmax_grad_norm = 0.5\nproximal_ewma = true\nproximal_ewma_com = 256.0\n\n[wandb]\nenabled = true\nproject = \"ahcrl-meta-__SLUG__\"\nname = \"__SLUG__-ppo-ewma-com256\"\ntags = [\"__SLUG__\", \"ppo-ewma\", \"ewma\"]\n\n[model]\nchannels = 128\nblocks = 4\n""",
            slug=slug,
            directory=directory,
        ),
        f"contests/{directory}/eval/.gitignore": "a.out\npahcer/\n",
        f"contests/{directory}/eval/main.cpp": """// Generated TorchScript submission will replace this file.\n#include <iostream>\n\nint main() {\n    // Empty output is a valid pipeline smoke test; implement the solver later.\n    return 0;\n}\n""",
        f"contests/{directory}/eval/pahcer_config.toml": render(
            """[general]\nversion = \"0.3.1\"\n\n[problem]\nproblem_name = \"__SLUG__\"\nobjective = \"Max\"\nscore_regex = '(?m)^\\s*Score\\s*=\\s*(?P<score>\\d+)\\s*$'\n\n[test]\nstart_seed = 0\nend_seed = 100\nthreads = 0\nout_dir = \"./pahcer\"\n\n[[test.compile_steps]]\nprogram = \"bash\"\nargs = [\n  \"-lc\",\n  \"\"\"\nset -euo pipefail\nTORCH_FLAGS=\\\"$(uv run python3 -c 'import torch; from torch.utils.cpp_extension import include_paths, library_paths; print(\\\" \\\".join([\\\"-I\\\"+p for p in include_paths()] + [\\\"-L\\\"+p for p in library_paths()] + [\\\"-Wl,-rpath,\\\"+p for p in library_paths()]))')\\\"\nTORCH_ABI=\\\"$(uv run python3 -c 'import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)')\\\"\ng++ -std=c++20 -O2 -D_GLIBCXX_USE_CXX11_ABI=\\\"${TORCH_ABI}\\\" ${TORCH_FLAGS} main.cpp -ltorch -ltorch_cpu -lc10 -o a.out\n\"\"\",\n]\n\n[[test.test_steps]]\nprogram = \"./a.out\"\nstdin = \"../tools/in/{SEED04}.txt\"\nstdout = \"../tools/out/{SEED04}.txt\"\nstderr = \"../tools/err/{SEED04}.txt\"\nmeasure_time = true\n\n[[test.test_steps]]\nprogram = \"cargo\"\nargs = [\"run\", \"--bin\", \"vis\", \"--release\", \"--\", \"./in/{SEED04}.txt\", \"./out/{SEED04}.txt\"]\ncurrent_dir = \"../tools\"\nmeasure_time = false\n""",
            slug=slug,
            directory=directory,
        ),
        f"src/ahcrl/contests/{slug}/__init__.py": "",
        f"src/ahcrl/contests/{slug}/encoder.py": """\"\"\"Contest-specific observation encoder.\"\"\"\n\n# TODO: Define the exact observation layout shared by Python and C++.\n""",
        f"src/ahcrl/contests/{slug}/model.py": """\"\"\"Contest-specific Actor-Critic model.\"\"\"\n\n# TODO: Reuse blocks from ahcrl.nn and implement the policy/value heads.\n""",
        f"src/ahcrl/contests/{slug}/train_ppo.py": """\"\"\"Contest-specific PPO-EWMA entry point.\n\nUse :class:`ahcrl.envs.RustVecEnv` with ``cargo_server_command`` and this\ncontest's ``rl-tools/Cargo.toml``. Reuse configuration resolution, run state,\ncheckpoints, W&B, and standard metrics from :mod:`ahcrl.training`; keep only\nthe model, rollout details, and contest metrics in this module.\"\"\"\n\n# TODO: Implement the contest-specific PPO-EWMA loop after the Rust environment and model exist.\n""",
        f"contests/{directory}/scripts/README.md": """# Scripts\n\n`export_torchscript_submit.py` should be added after the observation encoder and C++\nstate transition logic are fixed. Use the AHC061/AHC063 exporters as references.\n""",
        f"contests/{directory}/rl-tools/Cargo.toml": render(
            """[package]\nname = \"__SLUG__-rl-tools\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nahcrl-env-core = { path = \"../../../crates/ahcrl-env-core\" }\nserde = { version = \"1\", features = [\"derive\"] }\nserde_json = \"1\"\ntools = { path = \"../tools\" }\n\n[profile.dev]\noverflow-checks = false\n\n[profile.test]\noverflow-checks = false\n""",
            slug=slug,
            directory=directory,
        ),
        f"contests/{directory}/rl-tools/src/lib.rs": render(
            """use ahcrl_env_core::{ContestEnv, EnvFactory, EnvSpec};\nuse serde_json::Value;\n\n/// Factory for the contest-specific environments served to Python.\npub struct __FACTORY__;\n\n/// One contest instance. Replace this placeholder with the simulator state.\npub struct __ENV__;\n\nimpl EnvFactory for __FACTORY__ {\n    type Env = __ENV__;\n\n    fn from_config(_config: Value) -> Result<Self, String> {\n        // TODO: Deserialize and validate contest-specific configuration.\n        Ok(Self)\n    }\n\n    fn spec(&self) -> EnvSpec {\n        // TODO: Return the observation and metric TensorSpec values shared with Python.\n        panic!(\"TODO: define the contest environment specification\")\n    }\n\n    fn create(&self, _seed: u64) -> Result<Self::Env, String> {\n        // TODO: Build a seeded simulator from the official tools.\n        Ok(__ENV__)\n    }\n}\n\nimpl ContestEnv for __ENV__ {\n    fn validate_action(&self, _action: u32) -> Result<(), String> {\n        Err(\"TODO: validate a contest action\".to_owned())\n    }\n\n    fn step(&mut self, _action: u32) -> Result<(), String> {\n        Err(\"TODO: advance the contest simulator\".to_owned())\n    }\n\n    fn reward(&self) -> f32 {\n        0.0\n    }\n\n    fn done(&self) -> bool {\n        false\n    }\n\n    fn score(&self) -> i64 {\n        0\n    }\n\n    fn write_observation(&self, _name: &str, _destination: &mut [u8]) -> Result<(), String> {\n        Err(\"TODO: write an observation tensor\".to_owned())\n    }\n\n    fn write_metric(&self, _name: &str, _destination: &mut [u8]) -> Result<(), String> {\n        Err(\"TODO: write a metric tensor\".to_owned())\n    }\n}\n""",
            slug=slug,
            directory=directory,
        ),
        f"contests/{directory}/rl-tools/src/bin/rl_env.rs": render(
            """use __SLUG___rl_tools::__FACTORY__;\n\nfn main() {\n    ahcrl_env_core::server_main::<__FACTORY__>();\n}\n""",
            slug=slug,
            directory=directory,
        ),
    }
    files[f"contests/{directory}/rust-toolchain.toml"] = """[toolchain]
channel = "stable"
profile = "minimal"
"""
    files[f"contests/{directory}/rl-tools/IMPLEMENTATION.md"] = """# RL環境の実装順

`rl-tools/src/lib.rs` は「公式toolsをRL用プロトコルへ接続する層」である。公式の状態遷移・入力生成・採点をコピーして実装しない。

1. 公式toolsで `Input`、入力生成関数、状態型、状態遷移関数（`apply` / `step` など）、採点関数を探す。
2. `AhcXXXEnv::from_seed` で公式 generator から `Input` を作り、公式状態型を初期化する。固定入力で検証したい場合は `new(input)` も追加する。
3. 行動を `u32` の連番に割り当て、`validate_action` で範囲と合法性を確認する。action mask が必要なら観測に `mask: U8[action_count]` を追加する。
4. `step` は公式の状態遷移を一度だけ呼び、直前・直後の公式 score/cost から reward を作る。終了条件も公式のターン数・完了判定に合わせる。
5. `encode_*` で状態を固定shapeの tensor にし、`write_observation` から dtype に対応する helper で書き込む。GPU入力で転送量が支配的なら `F16` / `write_f16_slice` を検討する。
6. 複数 observation が同じ中間特徴を使う場合は、`prepare_observation(&mut self)` で1回だけ計算してcacheし、`write_observation` はcacheから書き込む。core は各 batch の直前にこの hook をenvごとに一度だけ呼ぶ。
7. 少数seedの軌跡について、RL環境の最終scoreと公式 scorer のscoreが一致するテストを書く。

## 公式toolsをどこまで編集してよいか

原則は無編集。状態の必要な値が private で読めない場合だけ、次の最小差分を入れる。

```rust
// tools/src/lib.rs
pub mod rl_bridge;
```

```rust
// tools/src/rl_bridge.rs
// 公式Stateの private field を借用で公開する StateView と state_view を置く。
// 状態変更用の関数は追加しない。
```

AHC063 はこの方式の実例で、`tools/src/rl_bridge.rs` が唯一の追加ファイルである。
"""
    files[f"contests/{directory}/README.md"] = (
        files[f"contests/{directory}/README.md"].replace(
            "3. `rl-tools/src/lib.rs` の `EnvFactory` / `ContestEnv` を、公式toolsを使う simulator・観測・action space・metricsで実装する。",
            "3. [rl-tools/IMPLEMENTATION.md](rl-tools/IMPLEMENTATION.md) の順に、公式simulator・行動空間・観測・報酬を接続する。",
        )
        + """\n## 公式toolsとの境界

`tools/` は公式スナップショットであり、状態遷移・採点・入力生成の再実装先ではない。RL環境は `rl-tools/` に置く。
公式状態の非公開フィールドを読む必要があるときだけ、`tools/src/rl_bridge.rs` と `tools/src/lib.rs` の `pub mod rl_bridge;` を追加してよい。bridge は読み取り専用のviewだけを公開し、状態遷移は必ず公式の `apply` / `step` を呼ぶ。

AHC063 の実例では、公式toolsからのソース差分はこの bridge と `pub mod rl_bridge;` だけである。次で確認できる。

```bash
git diff --no-index contests/ahc-063/eval/ahc063/tools/src/lib.rs contests/ahc-063/tools/src/lib.rs
git diff --no-index /dev/null contests/ahc-063/tools/src/rl_bridge.rs
```
"""
    )
    files[f"contests/{directory}/problem_en.md"] = (
        "# Problem Statement (English)\n\nTODO: AtCoder の英語問題文を保存する。\n"
    )
    files[f"src/ahcrl/contests/{slug}/encoder.py"] = (
        '"""コンテスト固有の観測エンコーダ。"""\n\n'
        "# TODO: Rust・Python・提出用 C++ で共有する観測レイアウトを定義する。\n"
    )
    files[f"src/ahcrl/contests/{slug}/model.py"] = (
        '"""コンテスト固有の Actor-Critic モデル。"""\n\n'
        "# TODO: ahcrl.nn の block を再利用して policy/value head を実装する。\n"
    )
    files[f"src/ahcrl/contests/{slug}/train_ppo.py"] = (
        '"""コンテスト固有の PPO-EWMA エントリポイント。"""\n\n'
        "# TODO: Rust環境とモデルの完成後、コンテスト固有の PPO-EWMA loop を実装する。\n"
    )
    files[f"contests/{directory}/scripts/README.md"] = """# Scripts

観測エンコーダと C++ の状態遷移が確定してから `export_torchscript_submit.py` を追加する。
AHC061/AHC063 の exporter を参考にする。
"""
    files[f"contests/{directory}/rl-tools/src/lib.rs"] = render(
        """use ahcrl_env_core::{ContestEnv, EnvFactory, EnvSpec, StepOutcome};
use serde::Deserialize;
use serde_json::Value;

/// Python に提供するコンテスト固有環境の factory。
pub struct __FACTORY__ {
    config: __ENV__Config,
}

/// 1問の環境。公式 simulator の状態をここに保持する。
pub struct __ENV__ {
    _seed: u64,
}

/// Python 側から渡すコンテスト固有設定。
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct __ENV__Config {}

impl EnvFactory for __FACTORY__ {
    type Env = __ENV__;

    fn from_config(config: Value) -> Result<Self, String> {
        // TODO: 設定項目を追加したら、ここで deserialize と検証を行う。
        let config = serde_json::from_value(config).map_err(|error| error.to_string())?;
        Ok(Self { config })
    }

    fn spec(&self) -> EnvSpec {
        // TODO: Python と共有する observation / metric の TensorSpec を返す。
        panic!("TODO: コンテスト環境の仕様を定義する")
    }

    fn create(&self, seed: u64) -> Result<Self::Env, String> {
        __ENV__::from_seed(seed, &self.config)
    }
}

impl __ENV__ {
    /// seed から公式入力を生成して1問の環境を作る。
    pub fn from_seed(seed: u64, _config: &__ENV__Config) -> Result<Self, String> {
        // TODO: `tools::gen(seed)` と公式 simulator の `new` を呼び、状態を初期化する。
        Ok(Self { _seed: seed })
    }
}

impl ContestEnv for __ENV__ {
    fn validate_action(&self, _action: u32) -> Result<(), String> {
        Err("TODO: コンテストの action を検証する".to_owned())
    }

    fn initial_outcome(&self) -> StepOutcome {
        StepOutcome {
            reward: 0.0,
            done: false,
            score: 0,
        }
    }

    fn step(&mut self, _action: u32) -> Result<StepOutcome, String> {
        Err("TODO: 公式 simulator を1ターン進める".to_owned())
    }

    fn write_observation(&self, _name: &str, _destination: &mut [u8]) -> Result<(), String> {
        Err("TODO: 観測 tensor を書き込む".to_owned())
    }

    fn write_metric(&self, _name: &str, _destination: &mut [u8]) -> Result<(), String> {
        Err("TODO: metric tensor を書き込む".to_owned())
    }
}
""",
        slug=slug,
        directory=directory,
    )
    for config_name in ("ppo_smoke.toml", "ppo_train.toml"):
        path = f"contests/{directory}/configs/{config_name}"
        files[path] = files[path].replace("\n", "\nenv_workers = 0\n", 1)
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contest", help="contest name, for example ahc068 or ahc-068")
    parser.add_argument(
        "--tools-url",
        default=None,
        help="explicit official tools zip URL; tools are not downloaded by default",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        slug, directory = normalize_contest(args.contest)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    contest_root = ROOT / "contests" / directory
    package_root = ROOT / "src" / "ahcrl" / "contests" / slug
    created: list[Path] = []

    for relative_path, content in starter_files(slug, directory).items():
        write_if_missing(ROOT / relative_path, content, created)

    for path in (
        contest_root / "tools",
        contest_root / "tools" / "in",
        contest_root / "tools" / "out",
        contest_root / "tools" / "err",
        contest_root / "artifacts" / "ppo",
        contest_root / "eval" / "pahcer",
        package_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    tools_url = args.tools_url
    tools_manifest = contest_root / "tools" / "Cargo.toml"
    if tools_manifest.exists():
        print(f"official tools: already present at {contest_root / 'tools'}")
    elif tools_url is None:
        print(
            f"official tools: not downloaded; place the official snapshot in "
            f"{contest_root / 'tools'} or pass --tools-url URL"
        )
    else:
        try:
            download_tools(tools_url, contest_root)
            print(f"official tools: downloaded from {tools_url}")
        except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as error:
            print(f"official tools: download failed: {error}", file=sys.stderr)
            print(f"  URL: {tools_url}", file=sys.stderr)
            print(f"  destination: {contest_root / 'tools'}", file=sys.stderr)

    print(f"initialized: {contest_root}")
    if created:
        print(f"created files: {len(created)}")
    else:
        print("created files: 0 (existing files were preserved)")
    print(f"next: cd {contest_root} && cargo build --release --manifest-path tools/Cargo.toml")
    print("next: make check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
