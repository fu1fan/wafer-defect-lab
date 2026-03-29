"""Download remote output directories, including large files, into local outputs/."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    DEFAULT_LOCAL_OUTPUT_ROOT,
    load_state,
    merge_deployment_overrides,
    relative_remote_output_path,
    resolve_run_state,
    state_exists,
    sync_output_tree,
)


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

    sync_output_tree(config, remote_dir, local_dir, max_size=None)
    print(f"Downloaded remote outputs to {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
