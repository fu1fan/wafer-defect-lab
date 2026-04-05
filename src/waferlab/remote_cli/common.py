"""Shared helpers for remote deployment and training CLIs."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = PROJECT_ROOT / ".waferlab" / "remote"
STATE_FILE = STATE_DIR / "state.json"
DEFAULT_LOCAL_REPORT_ROOT = PROJECT_ROOT / "outputs" / "remote"
DEFAULT_LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MINIMUM_SUPPORTED_CUDA = 12.6
MAXIMUM_SUPPORTED_CUDA = 13.0
DEFAULT_DEPLOYMENT_MODE = "system"


@dataclass(frozen=True)
class TorchSpec:
    torch: str
    torchvision: str
    index_url: str
    label: str


def _default_project_root() -> str:
    return "/workspace/wafer-defect-lab"


def _default_output_root(project_root: str) -> str:
    return f"{project_root}/outputs"


def _default_data_root(project_root: str) -> str:
    return f"{project_root}/data"


def _default_python_bin() -> str:
    return "python3"


def _default_bootstrap_cmd() -> str:
    return "{python_bin} -m pip install --no-cache-dir -r requirements.txt"


def _venv_python_bin() -> str:
    return "/workspace/waferlab-venv/bin/python"


def _venv_bootstrap_cmd() -> str:
    return (
        "python3 -m venv /workspace/waferlab-venv && "
        "/workspace/waferlab-venv/bin/pip install --no-cache-dir -r requirements.txt"
    )


TORCH_SPECS: tuple[tuple[float, TorchSpec], ...] = (
    (
        13.0,
        TorchSpec(
            torch="2.10.0",
            torchvision="0.25.0",
            index_url="https://download.pytorch.org/whl/cu130",
            label="CUDA 13.0",
        ),
    ),
    (
        12.8,
        TorchSpec(
            torch="2.10.0",
            torchvision="0.25.0",
            index_url="https://download.pytorch.org/whl/cu128",
            label="CUDA 12.8",
        ),
    ),
    (
        12.6,
        TorchSpec(
            torch="2.10.0",
            torchvision="0.25.0",
            index_url="https://download.pytorch.org/whl/cu126",
            label="CUDA 12.6",
        ),
    ),
    (
        12.4,
        TorchSpec(
            torch="2.6.0",
            torchvision="0.21.0",
            index_url="https://download.pytorch.org/whl/cu124",
            label="CUDA 12.4",
        ),
    ),
)

CPU_TORCH_SPEC = TorchSpec(
    torch="2.10.0",
    torchvision="0.25.0",
    index_url="https://download.pytorch.org/whl/cpu",
    label="CPU only",
)


@dataclass
class DeploymentConfig:
    host: str
    port: int = 22
    project_root: str = _default_project_root()
    data_root: str | None = None
    output_root: str | None = None
    deployment_mode: str = DEFAULT_DEPLOYMENT_MODE
    python_bin: str = _default_python_bin()
    bootstrap_cmd: str = _default_bootstrap_cmd()
    local_report_root: str = str(DEFAULT_LOCAL_REPORT_ROOT)

    def __post_init__(self) -> None:
        if self.deployment_mode not in {"system", "venv"}:
            raise ValueError("deployment_mode must be 'system' or 'venv'")
        if self.data_root is None:
            self.data_root = _default_data_root(self.project_root)
        if self.output_root is None:
            self.output_root = _default_output_root(self.project_root)

    @property
    def runs_root(self) -> str:
        return f"{self.output_root.rstrip('/')}/runs"


@dataclass
class RunState:
    run_id: str
    remote_run_dir: str
    local_report_dir: str
    config_path: str


@dataclass
class RemoteState:
    deployment: DeploymentConfig
    last_run: RunState | None = None


def generate_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def state_exists() -> bool:
    return STATE_FILE.exists()


def save_state(state: RemoteState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def load_state() -> RemoteState:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"No remote state found at {STATE_FILE}")
    payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    deployment_payload = dict(payload["deployment"])
    if "deployment_mode" not in deployment_payload:
        deployment_payload["deployment_mode"] = DEFAULT_DEPLOYMENT_MODE
        if deployment_payload.get("python_bin") == _venv_python_bin():
            deployment_payload["python_bin"] = _default_python_bin()
        if deployment_payload.get("bootstrap_cmd") == _venv_bootstrap_cmd():
            deployment_payload["bootstrap_cmd"] = _default_bootstrap_cmd()
    if deployment_payload.get("bootstrap_cmd") == "python3 -m pip install --no-cache-dir -r requirements.txt":
        deployment_payload["bootstrap_cmd"] = _default_bootstrap_cmd()
    deployment = DeploymentConfig(**deployment_payload)
    last_run = None
    if payload.get("last_run") is not None:
        last_run = RunState(**payload["last_run"])
    return RemoteState(deployment=deployment, last_run=last_run)


def merge_deployment_overrides(
    state: RemoteState | None,
    *,
    host: str | None,
    port: int | None,
    project_root: str | None,
    data_root: str | None,
    output_root: str | None,
    deployment_mode: str | None,
    python_bin: str | None,
    bootstrap_cmd: str | None,
    local_report_root: str | None,
) -> DeploymentConfig:
    base = state.deployment if state is not None else None
    mode = deployment_mode or (base.deployment_mode if base else DEFAULT_DEPLOYMENT_MODE)
    default_python_bin = _venv_python_bin() if mode == "venv" else _default_python_bin()
    default_bootstrap_cmd = _venv_bootstrap_cmd() if mode == "venv" else _default_bootstrap_cmd()
    return DeploymentConfig(
        host=host or (base.host if base else ""),
        port=port or (base.port if base else 22),
        project_root=project_root or (base.project_root if base else _default_project_root()),
        data_root=data_root or (base.data_root if base else None),
        output_root=output_root or (base.output_root if base else None),
        deployment_mode=mode,
        python_bin=python_bin or (base.python_bin if base and base.deployment_mode == mode else default_python_bin),
        bootstrap_cmd=bootstrap_cmd
        or (base.bootstrap_cmd if base and base.deployment_mode == mode else default_bootstrap_cmd),
        local_report_root=local_report_root
        or (base.local_report_root if base else str(DEFAULT_LOCAL_REPORT_ROOT)),
    )


def run_local(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def ssh_base(config: DeploymentConfig) -> list[str]:
    return [
        "ssh",
        "-p",
        str(config.port),
        "-o",
        "StrictHostKeyChecking=no",
        config.host,
    ]


def remote_run(
    config: DeploymentConfig,
    script: str,
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ssh_base(config) + [script]
    return run_local(command, check=check, capture_output=capture_output)


def detect_remote_cuda_version(config: DeploymentConfig) -> float | None:
    probe = remote_run(
        config,
        "bash -lc 'command -v nvidia-smi >/dev/null && nvidia-smi || true'",
        capture_output=True,
    )
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", probe.stdout)
    if not match:
        return None
    return float(match.group(1))


def select_torch_spec(cuda_version: float | None) -> TorchSpec:
    if cuda_version is None:
        return CPU_TORCH_SPEC
    if cuda_version < MINIMUM_SUPPORTED_CUDA:
        raise ValueError(
            f"Remote CUDA {cuda_version:.1f} is unsupported. "
            f"Minimum supported CUDA version is {MINIMUM_SUPPORTED_CUDA:.1f}."
        )
    for minimum_cuda, spec in TORCH_SPECS:
        if cuda_version >= minimum_cuda:
            return spec
    return CPU_TORCH_SPEC


def install_remote_torch(
    config: DeploymentConfig,
    spec: TorchSpec,
    *,
    python_bin: str | None = None,
) -> None:
    python_exe = python_bin or config.python_bin
    remote_run(
        config,
        (
            f"cd {q(config.project_root)} && "
            f"{shell_join([python_exe, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip'])} && "
            f"{shell_join([python_exe, '-m', 'pip', 'install', '--no-cache-dir', f'torch=={spec.torch}', f'torchvision=={spec.torchvision}', '--index-url', spec.index_url])}"
        ),
    )


def _rsync_ssh_transport(config: DeploymentConfig) -> str:
    return f"ssh -p {config.port} -o StrictHostKeyChecking=no"


def sync_project_code(config: DeploymentConfig) -> None:
    command = [
        "rsync",
        "-az",
        "--delete",
        "-e",
        _rsync_ssh_transport(config),
        "--exclude",
        "/.git/",
        "--exclude",
        "/.github/",
        "--exclude",
        "/.venv/",
        "--exclude",
        "__pycache__/",
        "--exclude",
        "/data/",
        "--exclude",
        "/outputs/",
        "--exclude",
        "/.waferlab/",
        f"{PROJECT_ROOT}/",
        f"{config.host}:{config.project_root}/",
    ]
    run_local(command)


def fetch_with_rsync(
    config: DeploymentConfig,
    remote_dir: str,
    local_dir: Path,
    *,
    includes: list[str],
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-avz",
        "-e",
        _rsync_ssh_transport(config),
    ]
    for pattern in includes:
        command.extend(["--include", pattern])
    command.extend(["--exclude", "*", f"{config.host}:{remote_dir.rstrip('/')}/", f"{local_dir}/"])
    run_local(command)


def sync_output_tree(
    config: DeploymentConfig,
    remote_dir: str,
    local_dir: Path,
    *,
    max_size: str | None = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-avz",
        "-e",
        _rsync_ssh_transport(config),
    ]
    if max_size:
        command.append(f"--max-size={max_size}")
    command.extend([f"{config.host}:{remote_dir.rstrip('/')}/", f"{local_dir}/"])
    run_local(command)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def q(value: str) -> str:
    return shlex.quote(value)


def ensure_remote_dirs(config: DeploymentConfig) -> None:
    remote_run(
        config,
        shell_join([
            "mkdir",
            "-p",
            config.project_root,
            config.data_root or _default_data_root(config.project_root),
            config.output_root or _default_output_root(config.project_root),
            config.runs_root,
        ]),
    )


def train_report_patterns() -> list[str]:
    return ["*/", "train.log", "history.json", "eval_metrics*.json", "*.png"]


def default_remote_env(config: DeploymentConfig) -> dict[str, str]:
    return {
        "PYTHONPATH": f"{config.project_root}/src",
        "WAFERLAB_DATA_ROOT": config.data_root or _default_data_root(config.project_root),
        "WAFERLAB_OUTPUT_ROOT": config.output_root or _default_output_root(config.project_root),
    }


def env_prefix(env_map: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env_map.items())


def resolve_run_state(
    state: RemoteState,
    *,
    run_id: str | None,
) -> RunState:
    if run_id is None:
        if state.last_run is None:
            raise ValueError("No previous remote run recorded. Pass --run-id explicitly.")
        return state.last_run
    remote_run_dir = f"{state.deployment.runs_root}/{run_id}"
    local_report_dir = str(Path(state.deployment.local_report_root) / run_id)
    config_path = (
        state.last_run.config_path
        if state.last_run
        else "configs/modal/experiments/wm811k_resnet18_baseline.yaml"
    )
    return RunState(
        run_id=run_id,
        remote_run_dir=remote_run_dir,
        local_report_dir=local_report_dir,
        config_path=config_path,
    )


def relative_remote_output_path(config: DeploymentConfig, remote_path: str) -> Path:
    output_root = PurePosixPath(config.output_root or _default_output_root(config.project_root))
    candidate = PurePosixPath(remote_path)
    try:
        relative = candidate.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Remote path {remote_path!r} is not inside remote output root {str(output_root)!r}.") from exc
    return Path(*relative.parts)


def print_command_summary(title: str, payload: dict[str, Any]) -> None:
    print(title)
    for key, value in payload.items():
        print(f"{key:16}: {value}")
