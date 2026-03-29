"""Download remote output directories, including large files, into local outputs/."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .common import (
    DEFAULT_LOCAL_OUTPUT_ROOT,
    load_state,
    merge_deployment_overrides,
    relative_remote_output_path,
    resolve_run_state,
    state_exists,
)


BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _blue(text: str) -> str:
    return f"{BLUE}{text}{RESET}"


def _green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def _print_tagged(tag: str, message: str) -> None:
    colorize = _green if tag == "remote" else _blue
    print(f"{colorize(f'[{tag}]')}{colorize(message)}")


def _stream_command(command: list[str], display_command: str) -> None:
    _print_tagged("runcmd", f" {display_command}")
    process = subprocess.Popen(
        command,
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
        raise subprocess.CalledProcessError(return_code, command)


def _sync_output_tree(config, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-avz",
        "-e",
        f"ssh -p {config.port} -o StrictHostKeyChecking=no",
        f"{config.host}:{remote_dir.rstrip('/')}/",
        f"{local_dir}/",
    ]
    display = " ".join(command)
    _stream_command(command, display)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all files for the latest or selected remote run, or mirror the whole remote outputs tree."
    )
    parser.add_argument("--host", help="Remote SSH host. Falls back to the saved deployment host.")
    parser.add_argument("--port", type=int, default=None, help="Remote SSH port")
    parser.add_argument("--run-id", default=None, help="Run id to fetch from. Defaults to the latest recorded run.")
    parser.add_argument(
        "--remote-subdir",
        default=None,
        help="Subdirectory under the remote outputs root to download, for example runs/run-20260330-120000",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the entire remote outputs tree instead of a single recorded run",
    )
    parser.add_argument(
        "--local-output-root",
        default=str(DEFAULT_LOCAL_OUTPUT_ROOT),
        help="Local outputs directory used as the sync destination",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not state_exists():
        raise SystemExit("No saved remote deployment found. Run scripts-remote/deploy.py first.")

    state = load_state()
    config = merge_deployment_overrides(
        state,
        host=args.host,
        port=args.port,
        project_root=None,
        data_root=None,
        output_root=None,
        deployment_mode=None,
        python_bin=None,
        bootstrap_cmd=None,
        local_report_root=None,
    )

    local_output_root = Path(args.local_output_root)

    if args.all:
        remote_dir = config.output_root
        local_dir = local_output_root
    elif args.remote_subdir is not None:
        remote_dir = f"{config.output_root.rstrip('/')}/{args.remote_subdir.strip('/')}"
        local_dir = local_output_root / args.remote_subdir.strip("/")
    else:
        run_state = resolve_run_state(state, run_id=args.run_id)
        remote_dir = run_state.remote_run_dir
        local_dir = local_output_root / relative_remote_output_path(config, remote_dir)

    _sync_output_tree(config, remote_dir, local_dir)
    print(f"Downloaded remote outputs to {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
