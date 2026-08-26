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
    return template.replace("__SLUG__", slug).replace("__DIRECTORY__", directory)


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
    return {
        f"contests/{directory}/AGENTS.md": f"""# {slug.upper()}\n\n問題文とコンテスト固有の制約を確認してから実装する。\n""",
        f"contests/{directory}/README.md": f"""# {slug.upper()}\n\n新しいAHCの作業ディレクトリ。\n\n- コンテスト: https://atcoder.jp/contests/{slug}\n- 問題文: [problem_ja.md](problem_ja.md) / [problem_en.md](problem_en.md)\n- 公式tools: `tools/`（取得後は原則編集しない）\n- PPO設定: `configs/`\n- Pahcer設定: `eval/pahcer_config.toml`\n- PPO成果物: `artifacts/ppo/`（git管理外）\n\n## 最初にすること\n\n1. `problem_ja.md` と `problem_en.md` に問題文を保存する。\n2. `tools/` にAtCoder公式toolsを配置し、`cargo build --release`する。\n3. `src/ahcrl/contests/{slug}/` のencoder・simulator・modelを実装する。\n4. `eval/main.cpp`を公式visで検証する。\n5. `make check`を通してからPPOを開始する。\n""",
        f"contests/{directory}/problem_ja.md": "# 問題文（日本語）\n\nTODO: AtCoderの問題文を保存する。\n",
        f"contests/{directory}/problem_en.md": "# Problem Statement (English)\n\nTODO: Save the AtCoder problem statement here.\n",
        f"contests/{directory}/configs/ppo_smoke.toml": render(
            """[train]\nnum_envs = 4\ntotal_steps = 4096\nrollout_steps = 32\nseed_start = 0\nseed_stride = 1\ndevice = \"cpu\"\ncompile = false\nlr = 0.0003\ngamma = 0.99\ngae_lambda = 0.95\nclip = 0.2\nepochs = 1\nminibatch_size = 32\nentropy_coef = 0.01\nvalue_coef = 0.5\nmax_grad_norm = 0.5\nartifact_dir = \"contests/__DIRECTORY__/artifacts/ppo\"\ncheckpoint_interval_updates = 1\nmodel_channels = 32\nmodel_blocks = 2\nmodel_block_type = \"convnext\"\nwandb_enabled = false\nwandb_project = \"ahcrl-meta-__SLUG__\"\nwandb_name = \"__SLUG__-ppo-smoke\"\nwandb_tags = [\"__SLUG__\", \"ppo\", \"smoke\"]\n""",
            slug=slug,
            directory=directory,
        ),
        f"contests/{directory}/configs/ppo_train.toml": render(
            """[train]\nnum_envs = 256\ntotal_steps = 20000000\nrollout_steps = 128\nseed_start = 0\nseed_stride = 1\ndevice = \"cuda\"\ncompile = true\nlr = 0.0003\ngamma = 0.99\ngae_lambda = 0.95\nclip = 0.2\nepochs = 1\nminibatch_size = 1024\nentropy_coef = 0.001\nvalue_coef = 0.5\nmax_grad_norm = 0.5\nartifact_dir = \"contests/__DIRECTORY__/artifacts/ppo\"\ncheckpoint_interval_updates = 40\nmodel_channels = 128\nmodel_blocks = 4\nmodel_block_type = \"convnext\"\nwandb_enabled = true\nwandb_project = \"ahcrl-meta-__SLUG__\"\nwandb_name = \"__SLUG__-ppo\"\nwandb_tags = [\"__SLUG__\", \"ppo\"]\n""",
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
        f"src/ahcrl/contests/{slug}/rust_vec_env.py": """\"\"\"Python wrapper for the contest-specific Rust vector environment.\"\"\"\n\n# TODO: Adapt the binary protocol from the existing contest implementation.\n""",
        f"src/ahcrl/contests/{slug}/train_ppo.py": """\"\"\"Contest-specific PPO entry point.\"\"\"\n\n# TODO: Reuse the PPO loop from ahc061 or ahc063 after implementing the environment.\n""",
        f"contests/{directory}/scripts/README.md": """# Scripts\n\n`export_torchscript_submit.py` should be added after the observation encoder and C++\nstate transition logic are fixed. Use the AHC061/AHC063 exporters as references.\n""",
        f"contests/{directory}/rl-tools/Cargo.toml": render(
            """[package]\nname = \"__SLUG__-rl-tools\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[profile.dev]\noverflow-checks = false\n\n[profile.test]\noverflow-checks = false\n""",
            slug=slug,
            directory=directory,
        ),
        f"contests/{directory}/rl-tools/src/lib.rs": """pub const BOARD_SIZE: usize = 20;\npub const MAX_OPERATIONS: usize = 100_000;\n\n// TODO: Add the simulator and action-space implementation.\n""",
        f"contests/{directory}/rl-tools/src/bin/rl_env.rs": """fn main() {\n    eprintln!(\"TODO: implement the contest-specific Rust RL environment\");\n}\n""",
    }


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
