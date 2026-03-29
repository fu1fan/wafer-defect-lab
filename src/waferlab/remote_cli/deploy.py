"""Deploy code, environment, and optional data prep to a remote shell host."""

from __future__ import annotations

import argparse

from .common import (
    MAXIMUM_SUPPORTED_CUDA,
    RemoteState,
    default_remote_env,
    detect_remote_cuda_version,
    ensure_remote_dirs,
    env_prefix,
    install_remote_torch,
    merge_deployment_overrides,
    print_command_summary,
    q,
    remote_run,
    save_state,
    select_torch_spec,
    shell_join,
    state_exists,
    sync_project_code,
    load_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy project code and environment to a remote shell host.")
    parser.add_argument("--host", help="Remote SSH host, e.g. root@1.2.3.4")
    parser.add_argument("--port", type=int, default=None, help="Remote SSH port")
    parser.add_argument("--project-root", default=None, help="Remote project directory")
    parser.add_argument("--data-root", default=None, help="Remote data root")
    parser.add_argument("--output-root", default=None, help="Remote output root")
    parser.add_argument("--python-bin", default=None, help="Remote Python executable")
    parser.add_argument("--bootstrap-cmd", default=None, help="Remote environment bootstrap command")
    parser.add_argument("--local-report-root", default=None, help="Local directory used for downloaded remote reports")
    parser.add_argument("--skip-torch-install", action="store_true", help="Do not auto-install torch/torchvision")
    parser.add_argument("--skip-sync", action="store_true", help="Do not rsync the local repo to the remote host")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Do not run the remote bootstrap command")
    parser.add_argument("--prepare-data", action="store_true", help="Run prepare_data.py on the remote host")
    parser.add_argument(
        "--dataset",
        default="WM-811K",
        help="Dataset name passed to prepare_data.py when --prepare-data is used",
    )
    parser.add_argument(
        "--process-subset",
        action="append",
        choices=["labeled", "unlabeled"],
        default=[],
        help="Subset(s) to build with process_data.py after prepare_data.py",
    )
    parser.add_argument("--force-data", action="store_true", help="Pass --force to prepare/process data steps")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prior_state = load_state() if state_exists() else None
    config = merge_deployment_overrides(
        prior_state,
        host=args.host,
        port=args.port,
        project_root=args.project_root,
        data_root=args.data_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
        bootstrap_cmd=args.bootstrap_cmd,
        local_report_root=args.local_report_root,
    )
    if not config.host:
        raise SystemExit("--host is required the first time you deploy to a remote machine.")

    print_command_summary(
        "Remote deployment",
        {
            "host": config.host,
            "port": config.port,
            "project_root": config.project_root,
            "data_root": config.data_root,
            "output_root": config.output_root,
            "python_bin": config.python_bin,
        },
    )

    ensure_remote_dirs(config)
    if not args.skip_sync:
        print(f"Syncing code to {config.host}:{config.project_root}")
        sync_project_code(config)

    if not args.skip_bootstrap:
        print("Bootstrapping remote environment")
        remote_run(
            config,
            f"cd {q(config.project_root)} && bash -lc {q(config.bootstrap_cmd)}",
        )
        if not args.skip_torch_install:
            cuda_version = detect_remote_cuda_version(config)
            torch_spec = select_torch_spec(cuda_version)
            if cuda_version is None:
                cuda_label = "no GPU detected"
            elif cuda_version > MAXIMUM_SUPPORTED_CUDA:
                cuda_label = (
                    f"CUDA {cuda_version:.1f} detected, "
                    f"capped to supported wheel set {torch_spec.label}"
                )
            else:
                cuda_label = torch_spec.label
            print(
                "Installing PyTorch for remote host: "
                f"{cuda_label} -> torch=={torch_spec.torch}, torchvision=={torch_spec.torchvision}"
            )
            install_remote_torch(config, torch_spec)

    env_map = default_remote_env(config)
    if args.prepare_data:
        force_flag = ["--force"] if args.force_data else []
        prepare_cmd = (
            f"cd {q(config.project_root)} && "
            f"{env_prefix(env_map)} {shell_join([config.python_bin, 'scripts/prepare_data.py', '--dataset', args.dataset, *force_flag])}"
        )
        print(f"Preparing data on {config.host}")
        remote_run(config, prepare_cmd)

    if args.process_subset:
        subset_flags: list[str] = []
        for subset in args.process_subset:
            subset_flags.extend(["--subset", subset])
        force_flag = ["--force"] if args.force_data else []
        process_cmd = (
            f"cd {q(config.project_root)} && "
            f"{env_prefix(env_map)} {shell_join([config.python_bin, 'scripts/process_data.py', *subset_flags, *force_flag])}"
        )
        print(f"Processing subsets on {config.host}: {', '.join(args.process_subset)}")
        remote_run(config, process_cmd)

    save_state(RemoteState(deployment=config, last_run=prior_state.last_run if prior_state else None))
    print("Remote deployment state saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
