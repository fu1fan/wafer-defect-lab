"""Shared helpers for remote deployment and training CLIs."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = PROJECT_ROOT / ".waferlab" / "remote"
STATE_FILE = STATE_DIR / "state.json"
DEFAULT_LOCAL_REPORT_ROOT = PROJECT_ROOT / "outputs" / "remote"


def _default_project_root() -> str:
    return "/workspace/wafer-defect-lab"


def _default_output_root(project_root: str) -> str:
    return f"{project_root}/outputs"


def _default_data_root(project_root: str) -> str:
    return f"{project_root}/data"


def _default_python_bin() -> str:
    return "/workspace/waferlab-venv/bin/python"


def _default_bootstrap_cmd() -> str:
    return (
        "python3 -m venv /workspace/waferlab-venv && "
        "/workspace/waferlab-venv/bin/pip install --no-cache-dir "
        "-r requirements.txt -r requirements-cu128.txt"
    )


@dataclass
class DeploymentConfig:
    host: str
    port: int = 22
    project_root: str = _default_project_root()
    data_root: str | None = None
    output_root: str | None = None
    python_bin: str = _default_python_bin()
    bootstrap_cmd: str = _default_bootstrap_cmd()
    local_report_root: str = str(DEFAULT_LOCAL_REPORT_ROOT)

    def __post_init__(self) -> None:
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
    deployment = DeploymentConfig(**payload["deployment"])
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
    python_bin: str | None,
    bootstrap_cmd: str | None,
    local_report_root: str | None,
) -> DeploymentConfig:
    base = state.deployment if state is not None else None
    return DeploymentConfig(
        host=host or (base.host if base else ""),
        port=port or (base.port if base else 22),
        project_root=project_root or (base.project_root if base else _default_project_root()),
        data_root=data_root or (base.data_root if base else None),
        output_root=output_root or (base.output_root if base else None),
        python_bin=python_bin or (base.python_bin if base else _default_python_bin()),
        bootstrap_cmd=bootstrap_cmd or (base.bootstrap_cmd if base else _default_bootstrap_cmd()),
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
    config_path = state.last_run.config_path if state.last_run else "configs/train/wm811k_resnet_baseline.yaml"
    return RunState(
        run_id=run_id,
        remote_run_dir=remote_run_dir,
        local_report_dir=local_report_dir,
        config_path=config_path,
    )


def print_command_summary(title: str, payload: dict[str, Any]) -> None:
    print(title)
    for key, value in payload.items():
        print(f"{key:16}: {value}")
