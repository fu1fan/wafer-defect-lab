"""Download remote checkpoint artifacts for a recorded remote run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import fetch_with_rsync, load_state, merge_deployment_overrides, resolve_run_state, state_exists


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download checkpoint files from the latest or selected remote run.")
    parser.add_argument("--host", help="Remote SSH host. Falls back to the saved deployment host.")
    parser.add_argument("--port", type=int, default=None, help="Remote SSH port")
    parser.add_argument("--run-id", default=None, help="Run id to fetch from. Defaults to the latest run.")
    parser.add_argument("--pattern", default="*.pt", help="Filename pattern to download")
    parser.add_argument("--local-report-root", default=None, help="Local root directory for remote reports")
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
        local_report_root=args.local_report_root,
    )
    run_state = resolve_run_state(state, run_id=args.run_id)
    local_dir = Path(config.local_report_root) / run_state.run_id
    fetch_with_rsync(
        config,
        run_state.remote_run_dir,
        local_dir,
        includes=["*/", args.pattern],
    )
    print(f"Weights downloaded to {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
