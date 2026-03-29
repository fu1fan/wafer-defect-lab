"""Run a remote training job on a shell-based GPU host and fetch reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    RemoteState,
    RunState,
    default_remote_env,
    env_prefix,
    fetch_with_rsync,
    generate_run_id,
    load_state,
    merge_deployment_overrides,
    print_command_summary,
    q,
    remote_run,
    save_state,
    shell_join,
    state_exists,
    train_report_patterns,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch training on a remote shell host and automatically download report artifacts."
    )
    parser.add_argument("--host", help="Remote SSH host. Falls back to the last deployed host.")
    parser.add_argument("--port", type=int, default=None, help="Remote SSH port")
    parser.add_argument("--project-root", default=None, help="Remote project directory")
    parser.add_argument("--data-root", default=None, help="Remote data root")
    parser.add_argument("--output-root", default=None, help="Remote output root")
    parser.add_argument("--python-bin", default=None, help="Remote Python executable")
    parser.add_argument("--local-report-root", default=None, help="Local directory for downloaded reports")
    parser.add_argument(
        "--config",
        default="configs/train/wm811k_resnet_baseline.yaml",
        help="Training config path relative to the project root",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run id")
    parser.add_argument(
        "--no-fetch-reports",
        action="store_true",
        help="Do not automatically rsync report files back after training",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args, extra_train_args = parser.parse_known_args()
    if extra_train_args[:1] == ["--"]:
        extra_train_args = extra_train_args[1:]
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
        python_bin=args.python_bin,
        bootstrap_cmd=None,
        local_report_root=args.local_report_root,
    )

    run_id = args.run_id or generate_run_id()
    remote_run_dir = f"{config.runs_root}/{run_id}"
    local_report_dir = Path(config.local_report_root) / run_id
    local_report_dir.mkdir(parents=True, exist_ok=True)

    train_args = ["--config", args.config, *extra_train_args]
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in train_args):
        train_args.extend(["--output-dir", remote_run_dir])

    env_map = default_remote_env(config)
    remote_script = (
        f"mkdir -p {q(remote_run_dir)} && "
        f"cd {q(config.project_root)} && "
        f"{env_prefix(env_map)} {shell_join([config.python_bin, 'scripts/train_classifier.py', *train_args])} "
        f"2>&1 | tee -a {q(f'{remote_run_dir}/train.log')}"
    )

    print_command_summary(
        "Remote training",
        {
            "host": config.host,
            "port": config.port,
            "config": args.config,
            "run_id": run_id,
            "remote_run_dir": remote_run_dir,
            "local_report_dir": local_report_dir,
        },
    )

    remote_run(config, remote_script)

    run_state = RunState(
        run_id=run_id,
        remote_run_dir=remote_run_dir,
        local_report_dir=str(local_report_dir),
        config_path=args.config,
    )
    save_state(RemoteState(deployment=config, last_run=run_state))

    if not args.no_fetch_reports:
        print("Downloading report artifacts")
        fetch_with_rsync(
            config,
            remote_run_dir,
            local_report_dir,
            includes=train_report_patterns(),
        )

    print("Remote training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
