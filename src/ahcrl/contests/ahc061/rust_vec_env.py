import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoder import BOARD_SIZE, NUM_PLANES

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
            text=False,
            bufsize=0,
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
            reward=obs["reward"],
            done=obs["done"],
            score=obs["score"],
        )

    def step_first_legal_noobs(self) -> None:
        self._send("STEP_FIRST_LEGAL_NOOBS")
        header = self._readline()
        if header != "OK_NOOBS":
            raise RuntimeError(f"unexpected rl_env noobs header: {header!r}")

    def bench_first_legal_internal(self, steps: int) -> tuple[int, float]:
        self._send(f"BENCH_FIRST_LEGAL_INTERNAL {steps}")
        header = self._readline()
        parts = header.split()
        if len(parts) != 3 or parts[0] != "OK_BENCH":
            raise RuntimeError(f"unexpected rl_env bench header: {header!r}")
        return int(parts[1]), float(parts[2])

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
        self.proc.stdin.write((line + "\n").encode("ascii"))
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("rl_env stdout is closed")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("rl_env stdout closed")
        return line.decode("ascii").strip()

    def _read_exact(self, size: int) -> bytes:
        if self.proc.stdout is None:
            raise RuntimeError("rl_env stdout is closed")
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) != size:
            raise RuntimeError(f"expected {size} bytes from rl_env, got {len(data)}")
        return data

    def _read_obs(self) -> dict[str, np.ndarray]:
        header = self._readline()
        if header.startswith("ERR"):
            raise RuntimeError(header)
        parts = header.split()
        if len(parts) != 5 or parts[0] != "OK":
            raise RuntimeError(f"unexpected rl_env header: {header!r}")
        nenv = int(parts[1])
        planes = int(parts[2])
        height = int(parts[3])
        width = int(parts[4])
        if nenv != self.num_envs:
            raise RuntimeError(f"unexpected env count {nenv}, expected {self.num_envs}")
        if (planes, height, width) != (NUM_PLANES, BOARD_SIZE, BOARD_SIZE):
            raise RuntimeError(
                f"unexpected encoded shape {(planes, height, width)}, "
                f"expected {(NUM_PLANES, BOARD_SIZE, BOARD_SIZE)}"
            )

        planes_size = nenv * NUM_PLANES * BOARD_SIZE * BOARD_SIZE * np.dtype("<f4").itemsize
        mask_size = nenv * BOARD_SIZE * BOARD_SIZE
        reward_size = nenv * np.dtype("<f4").itemsize
        done_size = nenv
        score_size = nenv * np.dtype("<i8").itemsize

        planes_arr = np.frombuffer(self._read_exact(planes_size), dtype="<f4").reshape(
            nenv,
            NUM_PLANES,
            BOARD_SIZE,
            BOARD_SIZE,
        )
        mask_arr = np.frombuffer(self._read_exact(mask_size), dtype=np.uint8).reshape(
            nenv,
            BOARD_SIZE * BOARD_SIZE,
        )
        reward_arr = np.frombuffer(self._read_exact(reward_size), dtype="<f4")
        done_arr = np.frombuffer(self._read_exact(done_size), dtype=np.uint8)
        score_arr = np.frombuffer(self._read_exact(score_size), dtype="<i8")
        end = self._readline()
        if end != "END":
            raise RuntimeError(f"expected END, got {end!r}")

        return {
            "planes": planes_arr.copy(),
            "mask": mask_arr.astype(bool),
            "reward": reward_arr.copy(),
            "done": done_arr.astype(bool),
            "score": score_arr.copy(),
        }
