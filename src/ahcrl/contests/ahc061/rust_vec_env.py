import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RL_TOOLS_MANIFEST = ROOT / "contests" / "ahc-061" / "rl-tools" / "Cargo.toml"


@dataclass
class StepResult:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    done: np.ndarray
    score: np.ndarray


class RustVecEnv:
    def __init__(
        self,
        num_envs: int,
        seed_start: int = 0,
        seed_stride: int = 1,
        fixed_m: int | None = None,
        fixed_u: int | None = None,
        release: bool = True,
    ) -> None:
        self.num_envs = num_envs
        cmd = ["cargo", "run"]
        if release:
            cmd.append("--release")
        cmd += ["--manifest-path", str(RL_TOOLS_MANIFEST), "--bin", "rl_env"]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._closed = False
        self.obs = self.reset(seed_start, seed_stride, fixed_m, fixed_u)

    def reset(
        self,
        seed_start: int = 0,
        seed_stride: int = 1,
        fixed_m: int | None = None,
        fixed_u: int | None = None,
    ) -> dict[str, np.ndarray]:
        m = 0 if fixed_m is None else fixed_m
        u = 0 if fixed_u is None else fixed_u
        self._send(f"RESET {self.num_envs} {seed_start} {seed_stride} {m} {u}")
        self.obs = self._read_obs()
        return self.obs

    def step(self, actions: np.ndarray) -> StepResult:
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions shape must be ({self.num_envs},), got {actions.shape}")
        self._send("STEP " + " ".join(str(int(a)) for a in actions))
        obs = self._read_obs()
        self.obs = obs
        return StepResult(
            obs=obs,
            reward=obs["reward"].astype(np.float32),
            done=obs["done"].astype(bool),
            score=obs["score"].astype(np.int64),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send("QUIT")
        except Exception:
            pass
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait(timeout=5)

    def _send(self, line: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("rl_env stdin is closed")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_obs(self) -> dict[str, np.ndarray]:
        if self.proc.stdout is None:
            raise RuntimeError("rl_env stdout is closed")
        header = self.proc.stdout.readline().strip()
        if header.startswith("ERR"):
            raise RuntimeError(header)
        parts = header.split()
        if len(parts) != 2 or parts[0] != "OK":
            raise RuntimeError(f"unexpected rl_env header: {header!r}")
        nenv = int(parts[1])
        obs = {
            "m": np.zeros(nenv, dtype=np.int64),
            "u": np.zeros(nenv, dtype=np.int64),
            "turn": np.zeros(nenv, dtype=np.int64),
            "done": np.zeros(nenv, dtype=np.int64),
            "score": np.zeros(nenv, dtype=np.int64),
            "n": np.zeros(nenv, dtype=np.int64),
            "reward": np.zeros(nenv, dtype=np.float32),
            "values": np.zeros((nenv, 100), dtype=np.float32),
            "owner": np.zeros((nenv, 100), dtype=np.int64),
            "level": np.zeros((nenv, 100), dtype=np.int64),
            "pos": np.zeros((nenv, 16), dtype=np.int64),
            "mask": np.zeros((nenv, 100), dtype=bool),
        }
        for expected_idx in range(nenv):
            env_line = self.proc.stdout.readline().strip().split()
            if len(env_line) != 9 or env_line[0] != "ENV":
                raise RuntimeError(f"bad ENV line: {' '.join(env_line)!r}")
            idx = int(env_line[1])
            if idx != expected_idx:
                raise RuntimeError(f"unexpected env index {idx}, expected {expected_idx}")
            obs["m"][idx] = int(env_line[2])
            obs["u"][idx] = int(env_line[3])
            obs["turn"][idx] = int(env_line[4])
            obs["done"][idx] = int(env_line[5])
            obs["score"][idx] = int(env_line[6])
            obs["n"][idx] = int(env_line[7])
            obs["reward"][idx] = float(env_line[8])
            obs["values"][idx] = _parse_line(
                self.proc.stdout.readline(), "VALUES", 100, np.float32
            )
            obs["owner"][idx] = _parse_line(
                self.proc.stdout.readline(), "OWNER", 100, np.int64
            )
            obs["level"][idx] = _parse_line(
                self.proc.stdout.readline(), "LEVEL", 100, np.int64
            )
            obs["pos"][idx] = _parse_line(self.proc.stdout.readline(), "POS", 16, np.int64)
            obs["mask"][idx] = _parse_line(
                self.proc.stdout.readline(), "MASK", 100, np.int64
            ).astype(
                bool
            )
        end = self.proc.stdout.readline().strip()
        if end != "END":
            raise RuntimeError(f"expected END, got {end!r}")
        return obs


def _parse_line(line: str, label: str, size: int, dtype: type[np.generic]) -> np.ndarray:
    parts = line.strip().split()
    if len(parts) != size + 1 or parts[0] != label:
        raise RuntimeError(f"bad {label} line")
    return np.asarray(parts[1:], dtype=dtype)
