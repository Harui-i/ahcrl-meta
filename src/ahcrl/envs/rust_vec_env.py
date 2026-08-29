from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

PROTOCOL_VERSION = 1
_DTYPES: dict[str, np.dtype] = {
    "f32": np.dtype("<f4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]

    @property
    def elements_per_env(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def bytes_per_env(self) -> int:
        return self.elements_per_env * self.dtype.itemsize


@dataclass
class StepResult:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    done: np.ndarray
    score: np.ndarray
    metrics: dict[str, np.ndarray]


def cargo_server_command(
    manifest_path: Path,
    *,
    binary: str = "rl_env",
    release: bool = True,
) -> list[str]:
    command = ["cargo", "run", "--quiet", "--locked"]
    if release:
        command.append("--release")
    command += ["--manifest-path", str(manifest_path), "--bin", binary]
    return command


class RustVecEnv:
    """Generic vector environment backed by one Rust child process.

    Observation arrays are views into a response buffer and remain valid only
    until the next reset or step. Rewards, dones, scores, and metrics are copied.
    """

    def __init__(
        self,
        command: Sequence[str],
        num_envs: int,
        *,
        config: dict[str, Any] | None = None,
        seed_start: int = 0,
        seed_stride: int = 1,
        cwd: Path | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not command:
            raise ValueError("command must not be empty")
        self.num_envs = num_envs
        self._closed = False
        self._buffer = bytearray()
        self._proc = subprocess.Popen(
            list(command),
            cwd=None if cwd is None else str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=False,
            bufsize=0,
        )
        try:
            self.observation_specs, self.metric_specs = self._initialize(config or {})
            self._batch_size = self._expected_batch_size()
            self._buffer = bytearray(self._batch_size)
            self.obs = self.reset(seed_start, seed_stride)
        except BaseException:
            self._terminate()
            raise

    def reset(self, seed_start: int = 0, seed_stride: int = 1) -> dict[str, np.ndarray]:
        self._validate_seed(seed_start, "seed_start")
        self._validate_seed(seed_stride, "seed_stride")
        self._send_line(f"RESET_ALL {seed_start} {seed_stride}")
        result = self._read_batch()
        self.obs = result.obs
        return self.obs

    def reset_done(
        self,
        done: np.ndarray,
        seed_start: int = 0,
        seed_stride: int = 1,
    ) -> dict[str, np.ndarray]:
        self._validate_seed(seed_start, "seed_start")
        self._validate_seed(seed_stride, "seed_stride")
        mask = np.asarray(done, dtype=np.bool_)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"done must have shape ({self.num_envs},), got {mask.shape}")
        self._send_line(f"RESET_MASK {seed_start} {seed_stride}", mask.view(np.uint8))
        result = self._read_batch()
        self.obs = result.obs
        return self.obs

    def step(self, actions: np.ndarray) -> StepResult:
        values = np.asarray(actions)
        if values.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},), got {values.shape}")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"actions must have an integer dtype, got {values.dtype}")
        if values.size:
            minimum = int(values.min())
            maximum = int(values.max())
            if minimum < 0 or maximum > np.iinfo(np.uint32).max:
                raise ValueError("actions must fit in uint32")
        encoded = np.asarray(values, dtype="<u4", order="C")
        self._send_line("STEP", encoded.view(np.uint8))
        result = self._read_batch()
        self.obs = result.obs
        return result

    def step_mask(self, active: np.ndarray, actions: np.ndarray) -> StepResult:
        """Advance only environments selected by ``active``.

        Actions for inactive environments are ignored by the server and need
        not be legal. This lets a finite evaluation batch retain environments
        that have already reached ``done`` while the remaining ones finish.
        """
        mask = np.asarray(active, dtype=np.bool_)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"active must have shape ({self.num_envs},), got {mask.shape}")
        values = np.asarray(actions)
        if values.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},), got {values.shape}")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"actions must have an integer dtype, got {values.dtype}")
        selected = values[mask]
        if selected.size:
            minimum = int(selected.min())
            maximum = int(selected.max())
            if minimum < 0 or maximum > np.iinfo(np.uint32).max:
                raise ValueError("active actions must fit in uint32")
        encoded = np.asarray(values, dtype="<u4", order="C")
        payload = mask.view(np.uint8).tobytes() + encoded.tobytes()
        self._send_line("STEP_MASK", payload)
        result = self._read_batch()
        self.obs = result.obs
        return result

    def visualizer_data(self) -> list[tuple[str, str]]:
        """現在の各 environment の可視化用 input / output テキストを返す。"""
        self._send_line("VISUALIZER_DATA")
        header = self._readline()
        self._raise_if_error(header)
        parts = header.split()
        if len(parts) != 2 or parts[0] != "OK_VISUALIZER_DATA":
            raise RuntimeError(f"unexpected VISUALIZER_DATA response: {header!r}")
        try:
            length = int(parts[1])
        except ValueError as error:
            raise RuntimeError(f"invalid visualizer data length in {header!r}") from error
        try:
            raw = json.loads(self._read_exact(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("invalid visualizer data payload") from error
        if self._read_exact(1) != b"\n":
            raise RuntimeError("visualizer data was not newline terminated")
        if not isinstance(raw, list) or len(raw) != self.num_envs:
            raise RuntimeError("visualizer data has an invalid environment count")
        data: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise RuntimeError("visualizer data item must be an object")
            input_text = item.get("input")
            output_text = item.get("output")
            if not isinstance(input_text, str) or not isinstance(output_text, str):
                raise RuntimeError("visualizer data item must contain string input and output")
            data.append((input_text, output_text))
        return data

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._proc.poll() is None:
                self._send_line("QUIT")
                response = self._readline()
                if response != "OK_QUIT":
                    raise RuntimeError(f"unexpected QUIT response: {response!r}")
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except BaseException:
            self._terminate()
            raise
        finally:
            self._closed = True

    def __enter__(self) -> RustVecEnv:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _initialize(self, config: dict[str, Any]) -> tuple[list[TensorSpec], list[TensorSpec]]:
        request = json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "num_envs": self.num_envs,
                "config": config,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_line(f"INIT {len(request)}", request)
        header = self._readline()
        self._raise_if_error(header)
        parts = header.split()
        if len(parts) != 2 or parts[0] != "OK_SPEC":
            raise RuntimeError(f"unexpected INIT response: {header!r}")
        try:
            length = int(parts[1])
        except ValueError as error:
            raise RuntimeError(f"invalid schema length in {header!r}") from error
        schema = json.loads(self._read_exact(length).decode("utf-8"))
        if self._read_exact(1) != b"\n":
            raise RuntimeError("schema was not newline terminated")
        if schema.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"unsupported schema protocol version: {schema!r}")
        observations = self._parse_specs(schema.get("observations"), "observations")
        metrics = self._parse_specs(schema.get("metrics"), "metrics")
        names = [spec.name for spec in observations + metrics]
        if len(names) != len(set(names)):
            raise RuntimeError("schema contains duplicate tensor names")
        if set(names) & {"reward", "done", "score"}:
            raise RuntimeError("schema uses a reserved tensor name")
        if not observations:
            raise RuntimeError("schema must contain at least one observation")
        return observations, metrics

    def _parse_specs(self, value: Any, field: str) -> list[TensorSpec]:
        if not isinstance(value, list):
            raise RuntimeError(f"schema field {field!r} must be a list")
        output: list[TensorSpec] = []
        for raw in value:
            if not isinstance(raw, dict):
                raise RuntimeError(f"invalid tensor spec in {field}: {raw!r}")
            name = raw.get("name")
            dtype_name = raw.get("dtype")
            shape = raw.get("shape")
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"invalid tensor name in {raw!r}")
            if dtype_name not in _DTYPES:
                raise RuntimeError(f"unsupported dtype in {raw!r}")
            if not isinstance(shape, list):
                raise RuntimeError(f"invalid tensor shape in {raw!r}")
            if any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape):
                raise RuntimeError(f"invalid tensor dimension in {raw!r}")
            output.append(TensorSpec(name, _DTYPES[dtype_name], tuple(shape)))
        return output

    def _expected_batch_size(self) -> int:
        tensors = sum(
            spec.bytes_per_env * self.num_envs
            for spec in self.observation_specs + self.metric_specs
        )
        required = self.num_envs * (np.dtype("<f4").itemsize + 1 + np.dtype("<i8").itemsize)
        return tensors + required

    def _read_batch(self) -> StepResult:
        header = self._readline()
        self._raise_if_error(header)
        parts = header.split()
        if len(parts) != 2 or parts[0] != "OK_BATCH":
            raise RuntimeError(f"unexpected batch response: {header!r}")
        try:
            length = int(parts[1])
        except ValueError as error:
            raise RuntimeError(f"invalid batch length in {header!r}") from error
        if length != self._batch_size:
            raise RuntimeError(f"batch has {length} bytes, expected {self._batch_size}")
        self._read_exact_into(memoryview(self._buffer))
        if self._read_exact(5) != b"\nEND\n":
            raise RuntimeError("batch was not terminated by END")

        offset = 0
        observations: dict[str, np.ndarray] = {}
        for spec in self.observation_specs:
            array, offset = self._array_from_buffer(spec, offset)
            observations[spec.name] = array.view(np.bool_) if spec.name == "mask" else array
        reward_bytes = self.num_envs * np.dtype("<f4").itemsize
        reward = np.frombuffer(self._buffer, dtype="<f4", count=self.num_envs, offset=offset).copy()
        offset += reward_bytes
        done = np.frombuffer(self._buffer, dtype=np.uint8, count=self.num_envs, offset=offset)
        done = done.view(np.bool_).copy()
        offset += self.num_envs
        score_bytes = self.num_envs * np.dtype("<i8").itemsize
        score = np.frombuffer(self._buffer, dtype="<i8", count=self.num_envs, offset=offset).copy()
        offset += score_bytes
        metrics: dict[str, np.ndarray] = {}
        for spec in self.metric_specs:
            array, offset = self._array_from_buffer(spec, offset)
            metrics[spec.name] = array.copy()
        if offset != self._batch_size:
            raise RuntimeError(f"decoded {offset} batch bytes, expected {self._batch_size}")
        return StepResult(observations, reward, done, score, metrics)

    def _array_from_buffer(self, spec: TensorSpec, offset: int) -> tuple[np.ndarray, int]:
        count = spec.elements_per_env * self.num_envs
        array = np.frombuffer(self._buffer, dtype=spec.dtype, count=count, offset=offset)
        array = array.reshape(self.num_envs, *spec.shape)
        return array, offset + count * spec.dtype.itemsize

    def _send_line(
        self, line: str, payload: bytes | bytearray | memoryview | np.ndarray = b""
    ) -> None:
        if self._closed:
            raise RuntimeError("RustVecEnv is closed")
        stdin = self._stdin()
        try:
            stdin.write((line + "\n").encode("ascii"))
            if isinstance(payload, np.ndarray):
                stdin.write(memoryview(payload))
            else:
                stdin.write(payload)
            stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(self._process_failure("failed to write to Rust env")) from error

    def _readline(self) -> str:
        try:
            line = self._stdout().readline()
        except OSError as error:
            raise RuntimeError(self._process_failure("failed to read from Rust env")) from error
        if not line:
            raise RuntimeError(self._process_failure("Rust env stdout closed"))
        return line.decode("utf-8").rstrip("\r\n")

    def _read_exact(self, size: int) -> bytes:
        output = bytearray(size)
        self._read_exact_into(memoryview(output))
        return bytes(output)

    def _read_exact_into(self, destination: memoryview) -> None:
        stdout = self._stdout()
        offset = 0
        while offset < len(destination):
            read = stdout.readinto(destination[offset:])
            if not read:
                raise RuntimeError(self._process_failure("Rust env stdout closed mid-payload"))
            offset += read

    def _raise_if_error(self, header: str) -> None:
        if header == "ERR" or header.startswith("ERR "):
            raise RuntimeError(header[4:] if len(header) > 4 else "Rust env reported an error")

    def _stdin(self) -> io.FileIO:
        if self._proc.stdin is None:
            raise RuntimeError("Rust env stdin is unavailable")
        return cast(io.FileIO, self._proc.stdin)

    def _stdout(self) -> io.FileIO:
        if self._proc.stdout is None:
            raise RuntimeError("Rust env stdout is unavailable")
        return cast(io.FileIO, self._proc.stdout)

    def _process_failure(self, message: str) -> str:
        returncode = self._proc.poll()
        return message if returncode is None else f"{message} (exit code {returncode})"

    def _terminate(self) -> None:
        self._closed = True
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)

    @staticmethod
    def _validate_seed(value: int, name: str) -> None:
        if not isinstance(value, int) or not 0 <= value <= np.iinfo(np.uint64).max:
            raise ValueError(f"{name} must fit in uint64")
