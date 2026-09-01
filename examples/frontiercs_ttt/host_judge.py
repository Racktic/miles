"""Start/stop an isolated Frontier-CS judge on Docker-less compute nodes.

The official Docker image exposes testlib inside the container. A host-mode
go-judge namespace may not see the benchmark checkout, so this manager stages a
private copy of the Node source under a configurable temporary directory and
sends testlib.h through go-judge's copyIn API. Repository sources are never
edited.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _default_frontiercs_root() -> str | None:
    configured = os.environ.get("FRONTIERCS_ROOT")
    if configured:
        return configured
    sibling = Path(__file__).resolve().parents[2].parent / "Frontier-CS"
    return str(sibling) if (sibling / "algorithmic" / "problems").is_dir() else None


def _state_base(configured: str | Path | None = None) -> Path:
    value = configured or os.environ.get("FRONTIERCS_JUDGE_STATE_BASE")
    return Path(value or tempfile.gettempdir()).expanduser().resolve()


def _service_root(api_port: int, state_base: str | Path | None = None) -> Path:
    return _state_base(state_base) / f"frontiercs-judge-{os.getuid()}-{api_port}"


def _resolve_executable(
    configured: str | None,
    *,
    command: str,
    repository_fallback: Path,
) -> Path:
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured).expanduser()
        candidates.append(configured_path)
        configured_from_path = shutil.which(configured)
        if configured_from_path:
            candidates.append(Path(configured_from_path))
    candidates.append(repository_fallback)
    from_path = shutil.which(command)
    if from_path:
        candidates.append(Path(from_path))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"cannot locate executable {command!r}; configure its explicit path "
        f"or install it on PATH (checked: {checked})"
    )


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return response.status == 200 and json.loads(response.read()).get("ok") is True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _gojudge_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/version", timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def _stop_pid(pid_path: Path, expected: tuple[str, ...]) -> None:
    pid = _pid(pid_path)
    if pid is None:
        return
    command = _cmdline(pid)
    if not command:
        return
    if not any(marker in command for marker in expected):
        raise RuntimeError(f"refusing to stop unexpected process {pid}: {command}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _cmdline(pid):
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)


def _patch_host_testlib(source: str) -> str:
    checker_before = """        const outName = 'chk';
        const res = await this.runOne({
            args: [CXX, srcName, '-O2', '-pipe', '-std=gnu++17', '-I', testlibPath, '-o', outName],
            env: SANDBOX_ENV,
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 64 }, { name: 'stderr', max: 1024 * 64 }],
            copyIn: { [srcName]: { content: checkerSourceText } },"""
    checker_after = """        const outName = 'chk';
        const testlibSource = await fs.readFile(path.join(testlibPath, 'testlib.h'), 'utf8');
        const res = await this.runOne({
            args: [CXX, srcName, '-O2', '-pipe', '-std=gnu++17', '-I', '.', '-o', outName],
            env: SANDBOX_ENV,
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 64 }, { name: 'stderr', max: 1024 * 64 }],
            copyIn: {
                [srcName]: { content: checkerSourceText },
                'testlib.h': { content: testlibSource }
            },"""
    interactor_before = """        const outName = 'interactor';
        const res = await this.runOne({
            args: [CXX, srcName, '-O2', '-pipe', '-std=gnu++17', '-I', testlibPath, '-o', outName],
            env: SANDBOX_ENV,
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 1024 }, { name: 'stderr', max: 1024 * 1024 }],
            copyIn: { [srcName]: { content: interactorSourceText } },"""
    interactor_after = """        const outName = 'interactor';
        const testlibSource = await fs.readFile(path.join(testlibPath, 'testlib.h'), 'utf8');
        const res = await this.runOne({
            args: [CXX, srcName, '-O2', '-pipe', '-std=gnu++17', '-I', '.', '-o', outName],
            env: SANDBOX_ENV,
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 1024 }, { name: 'stderr', max: 1024 * 1024 }],
            copyIn: {
                [srcName]: { content: interactorSourceText },
                'testlib.h': { content: testlibSource }
            },"""
    if source.count(checker_before) != 1 or source.count(interactor_before) != 1:
        raise RuntimeError("gojudge.js layout changed; refusing an ambiguous host-testlib patch")
    return source.replace(checker_before, checker_after).replace(
        interactor_before, interactor_after
    )


def _stage(root: Path, frontiercs_root: Path, node_modules: Path) -> None:
    algorithmic = frontiercs_root / "algorithmic"
    source_root = algorithmic / "judge" / "src"
    if not node_modules.is_dir():
        raise FileNotFoundError(
            f"missing Node dependencies at {node_modules}; run npm ci in {algorithmic} "
            "or pass --node-modules"
        )
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_root, root / "src")
    shutil.copy2(algorithmic / "server.js", root / "server.js")
    shutil.copy2(algorithmic / "package.json", root / "package.json")
    (root / "problems").symlink_to(algorithmic / "problems", target_is_directory=True)
    (root / "node_modules").symlink_to(node_modules, target_is_directory=True)
    (root / "data").mkdir()
    (root / "submissions").mkdir()
    (root / "gojudge-files").mkdir()
    gojudge_js = root / "src" / "gojudge.js"
    gojudge_js.write_text(
        _patch_host_testlib(gojudge_js.read_text(encoding="utf-8")), encoding="utf-8"
    )


def start(args: argparse.Namespace) -> None:
    if _healthy(args.api_port):
        print(f"judge already healthy at http://127.0.0.1:{args.api_port}")
        return
    if not args.frontiercs_root:
        raise ValueError(
            "cannot locate the Frontier-CS checkout; set FRONTIERCS_ROOT or pass "
            "--frontiercs-root"
        )
    frontiercs_root = Path(args.frontiercs_root).expanduser().resolve()
    if not (frontiercs_root / "algorithmic" / "problems").is_dir():
        raise FileNotFoundError(
            f"invalid Frontier-CS checkout (missing algorithmic/problems): {frontiercs_root}"
        )
    state_base = _state_base(args.state_base)
    root = _service_root(args.api_port, state_base)
    if root.exists():
        # Only remove our exact stale state root after checking its recorded PIDs.
        stop(
            argparse.Namespace(
                api_port=args.api_port,
                gojudge_port=args.gojudge_port,
                state_base=str(state_base),
                cleanup=True,
            )
        )
    node_modules = Path(
        args.node_modules or frontiercs_root / "algorithmic" / "node_modules"
    ).expanduser().resolve()
    gojudge_bin = _resolve_executable(
        args.gojudge_bin,
        command="go-judge",
        repository_fallback=frontiercs_root / "qwen_eval" / "go-judge",
    )
    init_bin = _resolve_executable(
        args.gojudge_init,
        command="go-judge-init",
        repository_fallback=frontiercs_root / "qwen_eval" / "go-judge-init",
    )
    node_bin = _resolve_executable(
        args.node_bin,
        command="node",
        repository_fallback=frontiercs_root / "qwen_eval" / "node20" / "bin" / "node",
    )
    _stage(root, frontiercs_root, node_modules)

    gojudge_log = (root / "gojudge.log").open("ab", buffering=0)
    gojudge = subprocess.Popen(
        [
            str(gojudge_bin),
            "-http-addr",
            f"127.0.0.1:{args.gojudge_port}",
            "-parallelism",
            str(args.parallelism),
            "-cgroup-prefix",
            f"gojudge{args.api_port}",
            "-dir",
            str(root / "gojudge-files"),
            "-container-init-path",
            str(init_bin),
        ],
        cwd=root,
        stdout=gojudge_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (root / "gojudge.pid").write_text(str(gojudge.pid), encoding="utf-8")
    for _ in range(80):
        if gojudge.poll() is not None:
            raise RuntimeError(f"go-judge exited; inspect {root / 'gojudge.log'}")
        if _gojudge_ready(args.gojudge_port):
            break
        time.sleep(0.25)
    else:
        raise TimeoutError(f"go-judge did not start; inspect {root / 'gojudge.log'}")

    environment = {
        **os.environ,
        "GJ_ADDR": f"http://127.0.0.1:{args.gojudge_port}",
        "PORT": str(args.api_port),
        "JUDGE_WORKERS": str(args.parallelism),
        "TESTLIB_INSIDE": str(frontiercs_root / "algorithmic" / "judge" / "include"),
        "SUBMISSIONS_DIR": str(root / "submissions"),
        "CANDIDATE_DIAGNOSTICS_MAX_CHARS": str(args.diagnostics_chars),
    }
    server_log = (root / "server.log").open("ab", buffering=0)
    server = subprocess.Popen(
        [str(node_bin), str(root / "server.js")],
        cwd=root,
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (root / "server.pid").write_text(str(server.pid), encoding="utf-8")
    for _ in range(80):
        if server.poll() is not None:
            raise RuntimeError(f"Node judge exited; inspect {root / 'server.log'}")
        if _healthy(args.api_port):
            print(f"judge_url=http://127.0.0.1:{args.api_port}")
            print(f"judge_state={root}")
            return
        time.sleep(0.25)
    raise TimeoutError(f"Node judge did not start; inspect {root / 'server.log'}")


def stop(args: argparse.Namespace) -> None:
    state_base = _state_base(args.state_base)
    root = _service_root(args.api_port, state_base)
    if not root.exists():
        return
    _stop_pid(root / "server.pid", (str(root / "server.js"),))
    _stop_pid(
        root / "gojudge.pid",
        (str(root / "gojudge-files"), f"127.0.0.1:{args.gojudge_port}"),
    )
    if args.cleanup:
        expected_parent = state_base
        if root.resolve().parent != expected_parent or not root.name.startswith(
            f"frontiercs-judge-{os.getuid()}-"
        ):
            raise RuntimeError(f"refusing unsafe cleanup target: {root}")
        shutil.rmtree(root)


def status(args: argparse.Namespace) -> None:
    root = _service_root(args.api_port, args.state_base)
    print(f"healthy={str(_healthy(args.api_port)).lower()}")
    print(f"judge_url=http://127.0.0.1:{args.api_port}")
    print(f"judge_state={root}")
    for name in ("server", "gojudge"):
        pid = _pid(root / f"{name}.pid")
        print(f"{name}_pid={pid or ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "stop", "status"))
    parser.add_argument("--frontiercs-root", default=_default_frontiercs_root())
    parser.add_argument("--api-port", type=int, default=8081)
    parser.add_argument("--gojudge-port", type=int, default=5050)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--diagnostics-chars", type=int, default=16 * 1024)
    parser.add_argument(
        "--state-base", default=os.environ.get("FRONTIERCS_JUDGE_STATE_BASE")
    )
    parser.add_argument(
        "--gojudge-bin", default=os.environ.get("FRONTIERCS_GOJUDGE_BIN")
    )
    parser.add_argument(
        "--gojudge-init", default=os.environ.get("FRONTIERCS_GOJUDGE_INIT")
    )
    parser.add_argument("--node-bin", default=os.environ.get("FRONTIERCS_NODE_BIN"))
    parser.add_argument(
        "--node-modules", default=os.environ.get("FRONTIERCS_NODE_MODULES")
    )
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.command == "start":
        start(args)
    elif args.command == "stop":
        stop(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
