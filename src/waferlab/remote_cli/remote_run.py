"""Sync code, run a remote script, and mirror small output artifacts locally."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .common import (
    DEFAULT_LOCAL_OUTPUT_ROOT,
    RemoteState,
    RunState,
    default_remote_env,
    env_prefix,
    generate_run_id,
    load_state,
    merge_deployment_overrides,
    print_command_summary,
    q,
    relative_remote_output_path,
    save_state,
    shell_join,
    ssh_base,
    state_exists,
    sync_output_tree,
    sync_project_code,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_CONFIG = "configs/train/wm811k_resnet_baseline.yaml"
BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync local code, run a script under scripts/ on the remote host, and sync small outputs back."
    )
    parser.add_argument("script", help="Script path under scripts/, for example scripts/train_classifier.py")
    parser.add_argument("--host", help="Remote SSH host. Falls back to the last deployed host.")
    parser.add_argument("--port", type=int, default=None, help="Remote SSH port")
    parser.add_argument("--project-root", default=None, help="Remote project directory")
    parser.add_argument("--data-root", default=None, help="Remote data root")
    parser.add_argument("--output-root", default=None, help="Remote output root")
    parser.add_argument("--python-bin", default=None, help="Remote Python executable")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id for train_classifier.py")
    parser.add_argument(
        "--local-output-root",
        default=str(DEFAULT_LOCAL_OUTPUT_ROOT),
        help="Local outputs directory that receives synced remote artifacts",
    )
    parser.add_argument(
        "--sync-max-size",
        default="32m",
        help="Only sync remote output files up to this size after the run, for example 16m or 500k",
    )
    parser.add_argument(
        "--no-sync-outputs",
        action="store_true",
        help="Do not sync small files from the remote outputs directory after the run",
    )
    return parser


def _validate_script(script: str) -> Path:
    script_path = Path(script)
    if script_path.is_absolute():
        raise SystemExit("Pass a path relative to the repo, such as scripts/train_classifier.py.")
    normalized = script_path.as_posix()
    if not normalized.startswith("scripts/"):
        raise SystemExit("Only scripts under scripts/ are supported.")
    local_path = PROJECT_ROOT / script_path
    if not local_path.is_file():
        raise SystemExit(f"Local script not found: {local_path}")
    return script_path


def _extract_arg_value(args: list[str], flag: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _blue(text: str) -> str:
    return f"{BLUE}{text}{RESET}"


def _green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def _print_tagged(tag: str, message: str) -> None:
    colorize = _green if tag == "remote" else _blue
    print(f"{colorize(f'[{tag}]')}{colorize(message)}")


def _stream_remote_command(config_host: str, ssh_command: list[str], remote_command: str) -> None:
    _print_tagged("runcmd", f" ssh {config_host} {remote_command}")
    process = subprocess.Popen(
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _print_tagged("remote", f" {line.rstrip()}")
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ssh_command)


def main() -> int:
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args()
    if passthrough_args[:1] == ["--"]:
        passthrough_args = passthrough_args[1:]

    script_path = _validate_script(args.script)
    prior_state = load_state() if state_exists() else None
    if prior_state is None and args.host is None:
        raise SystemExit("No saved remote deployment found. Run scripts-remote/deploy.py first or pass --host.")

    config = merge_deployment_overrides(
        prior_state,
        host=args.host,
        port=args.port,
        project_root=args.project_root,
        data_root=args.data_root,
        output_root=args.output_root,
        deployment_mode=None,
        python_bin=args.python_bin,
        bootstrap_cmd=None,
        local_report_root=None,
    )

    print_command_summary(
        "Remote run",
        {
            "host": config.host,
            "port": config.port,
            "script": script_path.as_posix(),
            "remote_project_root": config.project_root,
            "remote_output_root": config.output_root,
            "local_output_root": args.local_output_root,
            "sync_max_size": args.sync_max_size if not args.no_sync_outputs else "disabled",
        },
    )

    print(f"Syncing code to {config.host}:{config.project_root}")
    sync_project_code(config)

    run_state: RunState | None = None
    remote_script_path = script_path.as_posix()
    remote_script_args = list(passthrough_args)

    if remote_script_path == "scripts/train_classifier.py":
        run_id = args.run_id or generate_run_id()
        remote_run_dir = f"{config.runs_root}/{run_id}"
        if not _has_flag(remote_script_args, "--output-dir"):
            remote_script_args.extend(["--output-dir", remote_run_dir])
        run_state = RunState(
            run_id=run_id,
            remote_run_dir=remote_run_dir,
            local_report_dir=str(Path(args.local_output_root) / relative_remote_output_path(config, remote_run_dir)),
            config_path=_extract_arg_value(remote_script_args, "--config") or DEFAULT_TRAIN_CONFIG,
        )

    env_map = default_remote_env(config)
    env_map["PYTHONUNBUFFERED"] = "1"
    remote_command = (
        f"cd {q(config.project_root)} && "
        f"{env_prefix(env_map)} "
        f"{shell_join([config.python_bin, '-u', remote_script_path, *remote_script_args])}"
    )
    ssh_command = ssh_base(config) + [remote_command]
    _stream_remote_command(config.host, ssh_command, remote_command)

    if run_state is not None:
        save_state(RemoteState(deployment=config, last_run=run_state))
    elif prior_state is not None:
        save_state(RemoteState(deployment=config, last_run=prior_state.last_run))
    else:
        save_state(RemoteState(deployment=config, last_run=None))

    if not args.no_sync_outputs:
        remote_output_root = config.output_root
        if remote_output_root is None:
            raise SystemExit("Remote output root is not configured.")
        print(f"Syncing remote outputs <= {args.sync_max_size} into {args.local_output_root}")
        sync_output_tree(
            config,
            remote_output_root,
            Path(args.local_output_root),
            max_size=args.sync_max_size,
        )

    print("Remote run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
