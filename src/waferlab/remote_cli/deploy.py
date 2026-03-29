"""Deploy code, environment, and optional data prep to a remote shell host."""

from __future__ import annotations

import argparse
import json
import subprocess

from .common import (
    MAXIMUM_SUPPORTED_CUDA,
    PROJECT_ROOT,
    RemoteState,
    default_remote_env,
    env_prefix,
    merge_deployment_overrides,
    print_command_summary,
    q,
    save_state,
    select_torch_spec,
    shell_join,
    ssh_base,
    state_exists,
    load_state,
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


def _shell_render(parts: list[str]) -> str:
    return shell_join(parts)


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


def _run_remote(config, remote_command: str) -> None:
    ssh_command = ssh_base(config) + [remote_command]
    _stream_command(ssh_command, f"ssh {config.host} {remote_command}")


def _run_sync_project_code(config) -> None:
    command = [
        "rsync",
        "-az",
        "--delete",
        "-e",
        f"ssh -p {config.port} -o StrictHostKeyChecking=no",
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
    _stream_command(command, _shell_render(command))


def _ensure_remote_dirs(config) -> None:
    remote_command = _shell_render([
        "mkdir",
        "-p",
        config.project_root,
        config.data_root,
        config.output_root,
        config.runs_root,
    ])
    _run_remote(config, remote_command)


def _detect_remote_cuda_version(config) -> float | None:
    remote_command = "bash -lc 'command -v nvidia-smi >/dev/null && nvidia-smi || true'"
    ssh_command = ssh_base(config) + [remote_command]
    _print_tagged("runcmd", f" ssh {config.host} {remote_command}")
    process = subprocess.Popen(
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        stripped = line.rstrip()
        output_lines.append(stripped)
        _print_tagged("remote", f" {stripped}")
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ssh_command)
    combined = "\n".join(output_lines)
    import re

    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", combined)
    if not match:
        return None
    return float(match.group(1))


def _install_remote_torch(config, torch_spec) -> None:
    remote_command = (
        f"cd {q(config.project_root)} && "
        f"{_shell_render([config.python_bin, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip'])} && "
        f"{_shell_render([config.python_bin, '-m', 'pip', 'install', '--no-cache-dir', f'torch=={torch_spec.torch}', f'torchvision=={torch_spec.torchvision}', '--index-url', torch_spec.index_url])}"
    )
    _run_remote(config, remote_command)


def _resolve_bootstrap_command(config) -> str:
    if "{python_bin}" in config.bootstrap_cmd or "{project_root}" in config.bootstrap_cmd:
        return config.bootstrap_cmd.format(
            python_bin=config.python_bin,
            project_root=config.project_root,
        )
    if config.deployment_mode == "system" and config.bootstrap_cmd == "python3 -m pip install --no-cache-dir -r requirements.txt":
        return _shell_render([config.python_bin, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
    return config.bootstrap_cmd


def _report_remote_python_stack(config) -> None:
    probe_script = """
import platform
import sys

print("python=" + sys.executable)
print("python_version=" + platform.python_version())

try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda=" + str(torch.version.cuda))
    print("cuda_available=" + str(torch.cuda.is_available()))
except Exception as exc:
    print("torch=missing:" + exc.__class__.__name__ + ":" + str(exc))

try:
    import torchvision
    print("torchvision=" + torchvision.__version__)
except Exception as exc:
    print("torchvision=missing:" + exc.__class__.__name__ + ":" + str(exc))
""".strip()
    remote_command = (
        f"cd {q(config.project_root)} && "
        f"{_shell_render([config.python_bin, '-c', probe_script])}"
    )
    _run_remote(config, remote_command)


def _probe_bootstrap_python(config) -> str:
    bootstrap_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in (
        config.python_bin,
        "python3",
        "python",
        "/venv/main/bin/python3.12",
        "/venv/main/bin/python3",
        "/venv/bin/python3",
        "/venv/bin/python",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/conda/bin/python",
        "/opt/miniforge3/bin/python",
        "/opt/miniconda3/bin/python",
        "/root/miniconda3/bin/python",
        "/root/miniforge3/bin/python",
        "/workspace/.venv/bin/python",
        "/workspace/venv/bin/python",
    ):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        bootstrap_candidates.append(candidate)

    probe_script = (
        "set -eu\n"
        "for candidate in \\\n"
        + " \\\n".join(f"  {q(candidate)}" for candidate in bootstrap_candidates)
        + "\n"
        "do\n"
        "  if command -v \"$candidate\" >/dev/null 2>&1; then\n"
        "    command -v \"$candidate\"\n"
        "    exit 0\n"
        "  fi\n"
        "  if [ -x \"$candidate\" ]; then\n"
        "    printf '%s\\n' \"$candidate\"\n"
        "    exit 0\n"
        "  fi\n"
        "done\n"
        "exit 1"
    )
    remote_command = _shell_render(["sh", "-lc", probe_script])
    ssh_command = ssh_base(config) + [remote_command]
    _print_tagged("runcmd", f" ssh {config.host} {remote_command}")
    result = subprocess.run(
        ssh_command,
        text=True,
        capture_output=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        _print_tagged("remote", f" {line}")
    for line in result.stderr.splitlines():
        _print_tagged("remote", f" {line}")
    if result.returncode != 0:
        attempted = ", ".join(bootstrap_candidates)
        raise SystemExit(
            "Failed to find a usable remote Python interpreter for probing. "
            f"Tried: {attempted}. Pass --python-bin explicitly."
        )
    bootstrap_python = result.stdout.strip().splitlines()[-1].strip()
    if not bootstrap_python:
        attempted = ", ".join(bootstrap_candidates)
        raise SystemExit(
            "Remote Python probe bootstrap did not return an interpreter path. "
            f"Tried: {attempted}."
        )
    print(f"Bootstrap probe interpreter: {bootstrap_python}")
    return bootstrap_python


def _probe_remote_python_candidates(config) -> list[dict[str, str]]:
    probe_script = r"""
import json
import os
import pathlib
import shutil
import subprocess

candidates = []
seen = set()

def add_candidate(path, source):
    if not path:
        return
    real = os.path.realpath(path)
    if real in seen or not os.path.exists(real) or not os.access(real, os.X_OK):
        return
    seen.add(real)
    candidates.append((real, source))

def add_many(paths, source):
    for path in paths:
        add_candidate(path, source)

def run_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception:
        return ""

def collect_path_entries():
    path_map = {}
    commands = [
        ("env", ["bash", "-lc", "printf '%s\n' \"$PATH\""]),
        ("bash:lc", ["bash", "-lc", "bash -lc 'printf \"%s\n\" \"$PATH\"'"]),
        ("bash:ic", ["bash", "-lc", "bash -ic 'printf \"%s\n\" \"$PATH\"' || true"]),
        ("zsh:lc", ["bash", "-lc", "zsh -lc 'printf \"%s\n\" \"$PATH\"' 2>/dev/null || true"]),
        ("zsh:ic", ["bash", "-lc", "zsh -ic 'printf \"%s\n\" \"$PATH\"' 2>/dev/null || true"]),
    ]
    for source, cmd in commands:
        output = run_output(cmd)
        for line in output.splitlines():
            text = line.strip()
            if not text or "/" not in text or "duplicate session" in text or "open terminal failed" in text:
                continue
            if ":" not in text:
                continue
            path_map[source] = text
    return path_map

path_entries = collect_path_entries()
for source, path_value in path_entries.items():
    for entry in path_value.split(":"):
        entry = entry.strip()
        if not entry or not entry.startswith("/"):
            continue
        add_candidate(os.path.join(entry, "python"), f"path:{source}")
        add_candidate(os.path.join(entry, "python3"), f"path:{source}")
        add_candidate(os.path.join(entry, "conda"), f"path:{source}:conda")

for name in ("python3", "python"):
    resolved = shutil.which(name)
    if resolved:
        add_candidate(resolved, "system:path")
    output = run_output(["bash", "-lc", f"which -a {name} 2>/dev/null || true"])
    add_many([line.strip() for line in output.splitlines()], "system:path")

for extra in (
    "/opt/conda/bin/python",
    "/opt/miniforge3/bin/python",
    "/opt/miniforge/bin/python",
    "/opt/miniconda3/bin/python",
    "/opt/miniconda/bin/python",
    "/usr/bin/python3",
    "/usr/local/bin/python3",
    "/home/ubuntu/miniforge3/bin/python",
    "/home/ubuntu/miniconda3/bin/python",
    "/home/root/miniforge3/bin/python",
    "/home/root/miniconda3/bin/python",
    "/root/miniconda3/bin/python",
    "/root/anaconda3/bin/python",
    "/root/miniforge3/bin/python",
    "/root/mambaforge/bin/python",
    "/workspace/.venv/bin/python",
    "/workspace/venv/bin/python",
):
    source = "venv:common" if "/venv/" in extra or "/.venv/" in extra else "system:common"
    add_candidate(extra, source)

for conda_bin in (
    "conda",
    "/opt/conda/bin/conda",
    "/opt/miniforge3/bin/conda",
    "/opt/miniforge/bin/conda",
    "/opt/miniconda3/bin/conda",
    "/opt/miniconda/bin/conda",
    "/root/miniconda3/bin/conda",
    "/root/miniforge3/bin/conda",
    "/root/mambaforge/bin/conda",
    "/home/ubuntu/miniforge3/bin/conda",
    "/home/ubuntu/miniconda3/bin/conda",
):
    conda_json = run_output(["bash", "-lc", f"{conda_bin} env list --json 2>/dev/null || {conda_bin} info --envs --json 2>/dev/null || true"])
    if conda_json.strip():
        try:
            payload = json.loads(conda_json)
            for prefix in payload.get("envs", []):
                add_candidate(os.path.join(prefix, "bin", "python"), f"conda:{conda_bin}")
        except Exception:
            pass

for root in (
    pathlib.Path("/workspace"),
    pathlib.Path("/root"),
    pathlib.Path("/opt"),
    pathlib.Path("/home"),
    pathlib.Path.cwd(),
):
    if not root.exists():
        continue
    try:
        for pattern in (".venv/bin/python", "venv/bin/python", "env/bin/python"):
            for path in root.glob(f"**/{pattern}"):
                add_candidate(str(path), "venv:workspace")
    except Exception:
        pass

conda_json = run_output(["bash", "-lc", "conda env list --json 2>/dev/null || conda info --envs --json 2>/dev/null || true"])
if conda_json.strip():
    try:
        payload = json.loads(conda_json)
        for prefix in payload.get("envs", []):
            add_candidate(os.path.join(prefix, "bin", "python"), "conda")
    except Exception:
        pass

pyenv_root = run_output(["bash", "-lc", "pyenv root 2>/dev/null || true"]).strip()
if pyenv_root:
    versions_dir = pathlib.Path(pyenv_root) / "versions"
    if versions_dir.exists():
        for child in versions_dir.iterdir():
            add_candidate(str(child / "bin" / "python"), "pyenv")

poetry_envs = run_output(["bash", "-lc", "poetry env list --full-path 2>/dev/null || true"])
for line in poetry_envs.splitlines():
    env_path = line.strip().split()[0]
    add_candidate(os.path.join(env_path, "bin", "python"), "poetry")

project_poetry = run_output(["bash", "-lc", "cd /workspace/wafer-defect-lab 2>/dev/null && poetry env info -p 2>/dev/null || true"]).strip()
if project_poetry:
    add_candidate(os.path.join(project_poetry, "bin", "python"), "poetry:project")

project_pipenv = run_output(["bash", "-lc", "cd /workspace/wafer-defect-lab 2>/dev/null && pipenv --venv 2>/dev/null || true"]).strip()
if project_pipenv:
    add_candidate(os.path.join(project_pipenv, "bin", "python"), "pipenv:project")

for path, source in candidates:
    info = {
        "path": path,
        "source": source,
        "python_version": "",
        "torch": "missing",
        "torchvision": "missing",
        "torch_cuda": "",
        "cuda_available": "",
    }
    try:
        cmd = [
            path,
            "-c",
            (
                "import json, platform, sys\n"
                "payload={'path': sys.executable, 'python_version': platform.python_version()}\n"
                "try:\n"
                " import torch\n"
                " payload['torch']=torch.__version__\n"
                " payload['torch_cuda']=str(torch.version.cuda)\n"
                " payload['cuda_available']=str(torch.cuda.is_available())\n"
                "except Exception as exc:\n"
                " payload['torch']='missing:' + exc.__class__.__name__\n"
                "try:\n"
                " import torchvision\n"
                " payload['torchvision']=torchvision.__version__\n"
                "except Exception as exc:\n"
                " payload['torchvision']='missing:' + exc.__class__.__name__\n"
                "print(json.dumps(payload, ensure_ascii=True))\n"
            ),
        ]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
        info.update(json.loads(output.strip()))
    except Exception as exc:
        info["error"] = f"{exc.__class__.__name__}: {exc}"
    print(json.dumps(info, ensure_ascii=True))
""".strip()
    bootstrap_python = _probe_bootstrap_python(config)
    remote_command = _shell_render([bootstrap_python, "-c", probe_script])
    ssh_command = ssh_base(config) + [remote_command]
    _print_tagged("runcmd", f" ssh {config.host} {remote_command}")
    process = subprocess.Popen(
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    candidates: list[dict[str, str]] = []
    for line in process.stdout:
        stripped = line.rstrip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            _print_tagged("remote", f" {stripped}")
            continue
        candidates.append(payload)
        summary = (
            f"source={payload.get('source', '?')} "
            f"python={payload.get('path', '?')} "
            f"version={payload.get('python_version', '?')} "
            f"torch={payload.get('torch', '?')} "
            f"torchvision={payload.get('torchvision', '?')} "
            f"torch_cuda={payload.get('torch_cuda', '?')} "
            f"cuda_available={payload.get('cuda_available', '?')}"
        )
        if payload.get("error"):
            summary += f" error={payload['error']}"
        _print_tagged("remote", f" {summary}")
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ssh_command)
    return candidates


def _format_candidate(candidate: dict[str, str], index: int) -> str:
    return (
        f"{index}. source={candidate.get('source', '?')} "
        f"path={candidate.get('path', '?')} "
        f"python={candidate.get('python_version', '?')} "
        f"torch={candidate.get('torch', '?')} "
        f"torchvision={candidate.get('torchvision', '?')} "
        f"torch_cuda={candidate.get('torch_cuda', '?')} "
        f"cuda_available={candidate.get('cuda_available', '?')}"
    )


def _prompt_python_choice(candidates: list[dict[str, str]]) -> str:
    if not candidates:
        raise SystemExit("No remote Python interpreters were detected. Pass --python-bin explicitly.")
    print("Select remote python interpreter:")
    for index, candidate in enumerate(candidates, start=1):
        print(_format_candidate(candidate, index))
    while True:
        raw = input("Enter python index: ").strip()
        if not raw.isdigit():
            print("Please enter a numeric index.")
            continue
        choice = int(raw)
        if 1 <= choice <= len(candidates):
            selected = candidates[choice - 1]["path"]
            print(f"Selected remote python: {selected}")
            return selected
        print(f"Please enter a number between 1 and {len(candidates)}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy project code and environment to a remote shell host.")
    parser.add_argument("--host", help="Remote SSH host, e.g. root@1.2.3.4")
    parser.add_argument("--port", type=int, default=None, help="Remote SSH port")
    parser.add_argument("--project-root", default=None, help="Remote project directory")
    parser.add_argument("--data-root", default=None, help="Remote data root")
    parser.add_argument("--output-root", default=None, help="Remote output root")
    parser.add_argument(
        "--deployment-mode",
        choices=["system", "venv"],
        default=None,
        help="Remote Python setup mode. Defaults to system Python without a virtualenv.",
    )
    parser.add_argument("--python-bin", default=None, help="Remote Python executable")
    parser.add_argument("--bootstrap-cmd", default=None, help="Remote environment bootstrap command")
    parser.add_argument("--local-report-root", default=None, help="Local directory used for downloaded remote reports")
    parser.add_argument("--skip-torch-install", action="store_true", help="Do not auto-install torch/torchvision")
    parser.add_argument("--skip-sync", action="store_true", help="Do not rsync the local repo to the remote host")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Do not run the remote bootstrap command")
    parser.add_argument(
        "--probe-pythons-only",
        action="store_true",
        help="Only probe remote Python interpreters and exit without deploying",
    )
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
        deployment_mode=args.deployment_mode,
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
            "deployment_mode": config.deployment_mode,
            "python_bin": config.python_bin,
        },
    )

    print("Probing remote Python interpreters")
    candidates = _probe_remote_python_candidates(config)
    if args.probe_pythons_only:
        print("Python probe completed.")
        return 0

    if args.python_bin is not None:
        save_state(RemoteState(deployment=config, last_run=prior_state.last_run if prior_state else None))
        print(f"Persisted explicit remote python: {config.python_bin}")

    if config.deployment_mode == "system" and args.python_bin is None:
        selected_python = _prompt_python_choice(candidates)
        config.python_bin = selected_python
        save_state(RemoteState(deployment=config, last_run=prior_state.last_run if prior_state else None))
        print(f"Persisted selected remote python: {config.python_bin}")

    _ensure_remote_dirs(config)
    if not args.skip_sync:
        print(f"Syncing code to {config.host}:{config.project_root}")
        _run_sync_project_code(config)

    if not args.skip_bootstrap:
        print("Bootstrapping remote environment")
        bootstrap_command = _resolve_bootstrap_command(config)
        _run_remote(config, f"cd {q(config.project_root)} && bash -lc {q(bootstrap_command)}")
        print("Inspecting remote Python / Torch / CUDA stack")
        _report_remote_python_stack(config)
        if config.deployment_mode == "venv" and not args.skip_torch_install:
            cuda_version = _detect_remote_cuda_version(config)
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
            _install_remote_torch(config, torch_spec)
            print("Inspecting remote Python / Torch / CUDA stack after torch install")
            _report_remote_python_stack(config)

    env_map = default_remote_env(config)
    if args.prepare_data:
        force_flag = ["--force"] if args.force_data else []
        prepare_cmd = (
            f"cd {q(config.project_root)} && "
            f"{env_prefix(env_map)} {shell_join([config.python_bin, 'scripts/prepare_data.py', '--dataset', args.dataset, *force_flag])}"
        )
        print(f"Preparing data on {config.host}")
        _run_remote(config, prepare_cmd)

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
        _run_remote(config, process_cmd)

    save_state(RemoteState(deployment=config, last_run=prior_state.last_run if prior_state else None))
    print("Remote deployment state saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
