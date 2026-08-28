import subprocess
import tomllib
from pathlib import Path

import scripts.contest_init as contest_init
from ahcrl.training.config import load_toml_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULTS = {
    "num_envs": 1,
    "total_steps": 1,
    "rollout_steps": 1,
    "seed_start": 0,
    "seed_stride": 1,
    "device": "cpu",
    "compile": False,
    "artifact_dir": "artifacts",
    "checkpoint_interval_updates": 1,
    "lr": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "epochs": 1,
    "minibatch_size": 1,
    "entropy_coef": 0.0,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "wandb_enabled": False,
    "wandb_project": "project",
    "wandb_name": None,
    "wandb_tags": [],
    "model_channels": 1,
    "model_blocks": 1,
    "model_block_type": "convnext",
}


def test_normalize_contest_accepts_common_spellings() -> None:
    assert contest_init.normalize_contest("ahc068") == ("ahc068", "ahc-068")
    assert contest_init.normalize_contest("ahc-068") == ("ahc068", "ahc-068")
    assert contest_init.normalize_contest("68") == ("ahc068", "ahc-068")


def test_starter_pahcer_config_is_valid_toml() -> None:
    files = contest_init.starter_files("ahc068", "ahc-068")
    config = tomllib.loads(files["contests/ahc-068/eval/pahcer_config.toml"])

    assert config["problem"]["problem_name"] == "ahc068"
    assert config["problem"]["objective"] == "Max"
    assert len(config["test"]["test_steps"]) == 2
    assert config["test"]["test_steps"][1]["current_dir"] == "../tools"


def test_starter_ppo_configs_use_shared_training_sections(tmp_path: Path) -> None:
    files = contest_init.starter_files("ahc068", "ahc-068")

    for name in ("ppo_smoke.toml", "ppo_train.toml"):
        path = tmp_path / name
        path.write_text(files[f"contests/ahc-068/configs/{name}"], encoding="utf-8")
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        config = load_toml_config(path, defaults=CONFIG_DEFAULTS)

        assert "train" not in parsed
        assert set(parsed) == {"training", "ppo", "wandb", "model"}
        assert config["wandb_project"] == "ahcrl-meta-ahc068"
        assert config["model_block_type"] == "convnext"


def test_starter_uses_shared_rust_env_protocol(tmp_path: Path) -> None:
    files = contest_init.starter_files("ahc068", "ahc-068")
    manifest = files["contests/ahc-068/rl-tools/Cargo.toml"]
    library = files["contests/ahc-068/rl-tools/src/lib.rs"]
    server = files["contests/ahc-068/rl-tools/src/bin/rl_env.rs"]

    assert 'ahcrl-env-core = { path = "../../../crates/ahcrl-env-core" }' in manifest
    assert 'tools = { path = "../tools" }' in manifest
    assert "impl EnvFactory for Ahc068Factory" in library
    assert "impl ContestEnv for Ahc068Env" in library
    assert "server_main::<Ahc068Factory>()" in server
    assert "src/ahcrl/contests/ahc068/rust_vec_env.py" not in files

    library_path = tmp_path / "lib.rs"
    library_path.write_text(library, encoding="utf-8")
    subprocess.run(["rustfmt", "--check", str(library_path)], check=True)


def test_contest_check_make_target_accepts_generated_directory_name() -> None:
    result = subprocess.run(
        ["make", "-n", "contest-check", "CONTEST=ahc-068"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "contests/ahc-068/rl-tools/Cargo.toml" in result.stdout


def test_main_creates_layout_and_preserves_existing_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(contest_init, "ROOT", tmp_path)

    assert contest_init.main(["ahc068"]) == 0
    contest_root = tmp_path / "contests" / "ahc-068"
    assert (contest_root / "README.md").exists()
    assert (contest_root / "tools" / "in").is_dir()
    assert (contest_root / "tools" / "out").is_dir()
    assert not (contest_root / "eval" / "tools").exists()

    readme = contest_root / "README.md"
    readme.write_text("user content\n", encoding="utf-8")
    assert contest_init.main(["ahc068"]) == 0
    assert readme.read_text(encoding="utf-8") == "user content\n"
