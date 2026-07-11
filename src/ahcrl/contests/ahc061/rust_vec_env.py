import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from .encoder import BOARD_SIZE, CRITIC_FEATURE_SHAPE, CRITIC_FEATURE_SIZE, NUM_PLANES

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
        pf_particles: int = 16,
        release: bool = True,
    ) -> None:
        self.num_envs = num_envs
        self.pf_particles = pf_particles
        requested_workers = int(os.environ.get("AHC061_ENV_WORKERS", "0") or "0")
        self.num_workers = requested_workers if requested_workers > 0 else min(2, num_envs)
        self.num_workers = max(1, min(self.num_workers, num_envs))
        base = num_envs // self.num_workers
        rem = num_envs % self.num_workers
        self.worker_env_counts = [base + (1 if i < rem else 0) for i in range(self.num_workers)]
        self.worker_offsets = np.cumsum([0, *self.worker_env_counts[:-1]], dtype=np.int64)
        self._obs_buffers: dict[str, bytearray] = {}
        cmd = ["cargo", "run"]
        if release:
            cmd.append("--release")
        cmd += ["--manifest-path", str(RL_TOOLS_MANIFEST), "--bin", "rl_env"]
        self.procs = [self._start_proc(cmd, pf_particles) for _ in self.worker_env_counts]
        self._closed = False
        self.obs = self.reset(seed_start, seed_stride, fixed_m, fixed_u)

    def _buffer(self, name: str, size: int) -> bytearray:
        buf = self._obs_buffers.get(name)
        if buf is None or len(buf) != size:
            buf = bytearray(size)
            self._obs_buffers[name] = buf
        return buf

    def _start_proc(self, cmd: list[str], pf_particles: int) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        env["AHC061_PF_PARTICLES"] = str(pf_particles)
        env.setdefault("AHC061_ENCODE_THREADS", "16")
        return subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=False,
            bufsize=0,
        )

    def reset(
        self,
        seed_start: int = 0,
        seed_stride: int = 1,
        fixed_m: int | None = None,
        fixed_u: int | None = None,
    ) -> dict[str, np.ndarray]:
        m = 0 if fixed_m is None else fixed_m
        u = 0 if fixed_u is None else fixed_u
        for proc, count, offset in zip(
            self.procs,
            self.worker_env_counts,
            self.worker_offsets,
            strict=True,
        ):
            worker_seed_start = seed_start + int(offset) * seed_stride
            self._send(proc, f"RESET {count} {worker_seed_start} {seed_stride} {m} {u}")
        self.obs = self._read_obs_all()
        return self.obs

    def step(self, actions: np.ndarray) -> StepResult:
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions shape must be ({self.num_envs},), got {actions.shape}")
        start = 0
        for proc, count in zip(self.procs, self.worker_env_counts, strict=True):
            end = start + count
            self._send_actions_binary(proc, actions[start:end])
            start = end
        obs = self._read_obs_all()
        self.obs = obs
        return StepResult(
            obs=obs,
            reward=obs["reward"],
            done=obs["done"],
            score=obs["score"],
        )

    def step_first_legal_noobs(self) -> None:
        for proc in self.procs:
            self._send(proc, "STEP_FIRST_LEGAL_NOOBS")
        for proc in self.procs:
            header = self._readline(proc)
            if header != "OK_NOOBS":
                raise RuntimeError(f"unexpected rl_env noobs header: {header!r}")

    def bench_first_legal_internal(self, steps: int) -> tuple[int, float]:
        for proc in self.procs:
            self._send(proc, f"BENCH_FIRST_LEGAL_INTERNAL {steps}")
        total_env_steps = 0
        max_elapsed = 0.0
        for proc in self.procs:
            header = self._readline(proc)
            parts = header.split()
            if len(parts) != 3 or parts[0] != "OK_BENCH":
                raise RuntimeError(f"unexpected rl_env bench header: {header!r}")
            total_env_steps += int(parts[1])
            max_elapsed = max(max_elapsed, float(parts[2]))
        return total_env_steps, max_elapsed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for proc in self.procs:
            try:
                self._send(proc, "QUIT")
            except Exception:
                pass
        for proc in self.procs:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)

    def _send(self, proc: subprocess.Popen[bytes], line: str) -> None:
        if proc.stdin is None:
            raise RuntimeError("rl_env stdin is closed")
        proc.stdin.write((line + "\n").encode("ascii"))
        proc.stdin.flush()

    def _send_actions_binary(self, proc: subprocess.Popen[bytes], actions: np.ndarray) -> None:
        if proc.stdin is None:
            raise RuntimeError("rl_env stdin is closed")
        action_bytes = np.asarray(actions, dtype=np.uint8, order="C")
        proc.stdin.write(b"STEP_BIN\n")
        proc.stdin.write(memoryview(action_bytes))
        proc.stdin.flush()

    def _readline(self, proc: subprocess.Popen[bytes]) -> str:
        if proc.stdout is None:
            raise RuntimeError("rl_env stdout is closed")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("rl_env stdout closed")
        return line.decode("ascii").strip()

    def _read_obs_header(
        self,
        proc: subprocess.Popen[bytes],
        expected_num_envs: int,
    ) -> tuple[np.dtype, int]:
        header = self._readline(proc)
        if header.startswith("ERR"):
            raise RuntimeError(header)
        parts = header.split()
        if len(parts) != 6 or parts[0] not in {"OK", "OKF16"}:
            raise RuntimeError(f"unexpected rl_env header: {header!r}")
        planes_dtype = np.dtype("<f2") if parts[0] == "OKF16" else np.dtype("<f4")
        nenv = int(parts[1])
        planes = int(parts[2])
        height = int(parts[3])
        width = int(parts[4])
        critic_feature_size = int(parts[5])
        if nenv != expected_num_envs:
            raise RuntimeError(f"unexpected env count {nenv}, expected {expected_num_envs}")
        if (planes, height, width) != (NUM_PLANES, BOARD_SIZE, BOARD_SIZE):
            raise RuntimeError(
                f"unexpected encoded shape {(planes, height, width)}, "
                f"expected {(NUM_PLANES, BOARD_SIZE, BOARD_SIZE)}"
            )
        if critic_feature_size != CRITIC_FEATURE_SIZE:
            raise RuntimeError(
                f"unexpected critic feature size {critic_feature_size}, "
                f"expected {CRITIC_FEATURE_SIZE}"
            )
        return planes_dtype, critic_feature_size

    def _read_exact_into(self, proc: subprocess.Popen[bytes], view: memoryview) -> None:
        if proc.stdout is None:
            raise RuntimeError("rl_env stdout is closed")
        stdout = cast(io.FileIO, proc.stdout)
        remaining = len(view)
        offset = 0
        while remaining > 0:
            read_size = stdout.readinto(view[offset:])
            if not read_size:
                break
            offset += read_size
            remaining -= read_size
        if remaining != 0:
            raise RuntimeError(f"expected {len(view)} bytes from rl_env, got {offset}")

    def _read_obs_all(self) -> dict[str, np.ndarray]:
        plane_itemsize: int | None = None
        plane_dtype: np.dtype | None = None
        headers: list[tuple[subprocess.Popen[bytes], int, np.dtype, int]] = []
        for proc, count in zip(self.procs, self.worker_env_counts, strict=True):
            dtype, critic_feature_size = self._read_obs_header(proc, count)
            if plane_dtype is None:
                plane_dtype = dtype
                plane_itemsize = dtype.itemsize
            elif dtype != plane_dtype:
                raise RuntimeError(f"mixed plane dtypes from workers: {plane_dtype} and {dtype}")
            headers.append((proc, count, dtype, critic_feature_size))

        assert plane_dtype is not None
        assert plane_itemsize is not None
        per_env_planes_size = NUM_PLANES * BOARD_SIZE * BOARD_SIZE * plane_itemsize
        per_env_critic_feature_size = CRITIC_FEATURE_SIZE * plane_itemsize
        per_env_mask_size = BOARD_SIZE * BOARD_SIZE
        reward_itemsize = np.dtype("<f4").itemsize
        score_itemsize = np.dtype("<i8").itemsize

        planes_data = self._buffer("planes", self.num_envs * per_env_planes_size)
        posterior_data = self._buffer(
            "critic_posterior",
            self.num_envs * per_env_critic_feature_size,
        )
        oracle_data = self._buffer(
            "critic_oracle",
            self.num_envs * per_env_critic_feature_size,
        )
        mask_data = self._buffer("mask", self.num_envs * per_env_mask_size)
        reward_data = self._buffer("reward", self.num_envs * reward_itemsize)
        done_data = self._buffer("done", self.num_envs)
        score_data = self._buffer("score", self.num_envs * score_itemsize)

        planes_view = memoryview(planes_data)
        posterior_view = memoryview(posterior_data)
        oracle_view = memoryview(oracle_data)
        mask_view = memoryview(mask_data)
        reward_view = memoryview(reward_data)
        done_view = memoryview(done_data)
        score_view = memoryview(score_data)

        env_offset = 0
        for proc, count, _, _ in headers:
            plane_start = env_offset * per_env_planes_size
            plane_end = plane_start + count * per_env_planes_size
            critic_start = env_offset * per_env_critic_feature_size
            critic_end = critic_start + count * per_env_critic_feature_size
            mask_start = env_offset * per_env_mask_size
            mask_end = mask_start + count * per_env_mask_size
            reward_start = env_offset * reward_itemsize
            reward_end = reward_start + count * reward_itemsize
            score_start = env_offset * score_itemsize
            score_end = score_start + count * score_itemsize

            self._read_exact_into(proc, planes_view[plane_start:plane_end])
            self._read_exact_into(proc, posterior_view[critic_start:critic_end])
            self._read_exact_into(proc, oracle_view[critic_start:critic_end])
            self._read_exact_into(proc, mask_view[mask_start:mask_end])
            self._read_exact_into(proc, reward_view[reward_start:reward_end])
            self._read_exact_into(proc, done_view[env_offset : env_offset + count])
            self._read_exact_into(proc, score_view[score_start:score_end])
            env_offset += count

            end = self._readline(proc)
            if end != "END":
                raise RuntimeError(f"expected END, got {end!r}")

        return {
            "planes": np.frombuffer(planes_data, dtype=plane_dtype).reshape(
                self.num_envs,
                NUM_PLANES,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            "critic_posterior": np.frombuffer(posterior_data, dtype=plane_dtype).reshape(
                self.num_envs,
                *CRITIC_FEATURE_SHAPE,
            ),
            "critic_oracle": np.frombuffer(oracle_data, dtype=plane_dtype).reshape(
                self.num_envs,
                *CRITIC_FEATURE_SHAPE,
            ),
            "mask": np.frombuffer(mask_data, dtype=np.bool_).reshape(
                self.num_envs,
                BOARD_SIZE * BOARD_SIZE,
            ),
            "reward": np.frombuffer(reward_data, dtype="<f4"),
            "done": np.frombuffer(done_data, dtype=np.bool_),
            "score": np.frombuffer(score_data, dtype="<i8"),
        }
