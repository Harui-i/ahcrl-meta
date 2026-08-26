import tomllib

import scripts.contest_init as contest_init


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
