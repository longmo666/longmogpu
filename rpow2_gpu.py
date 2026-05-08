#!/usr/bin/env python3
"""
GPU protocol-level RPOW2 miner.

Python handles the API protocol:
  POST /challenge -> run CUDA nonce search -> POST /mint

The CUDA helper must be compiled first:
  nvcc -O3 -std=c++17 -o /root/rpow2_cuda_search /root/rpow2_cuda_search.cu
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://api.rpow2.com"
DEFAULT_COOKIE_FILE = Path("/root/.rpow2_cookie")
DEFAULT_CUDA_HELPER = Path("/root/rpow2_cuda_search")
DEFAULT_PID_FILE = Path("/root/rpow2_gpu_miner.pid")
DEFAULT_LOG_FILE = Path("/root/rpow2_gpu_miner.log")

COOKIE_ATTR_NAMES = {
    "domain",
    "expires",
    "httponly",
    "max-age",
    "path",
    "samesite",
    "secure",
}

RUNNING = True
ACTIVE_GPU_PROCESS: subprocess.Popen[str] | None = None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def redact(value: Any) -> str:
    text = str(value)
    text = re.sub(r"rpow_session=([^;\s]+)", "rpow_session=<redacted>", text, flags=re.I)
    text = re.sub(r"(cookie\s*[:=]\s*)([^\n]+)", r"\1<redacted>", text, flags=re.I)
    return text


class LiveLogger:
    def __init__(self) -> None:
        lane = os.environ.get("RPOW2_LANE", "").strip()
        self.prefix = f"[lane {lane}] " if lane else ""

    def log(self, message: str) -> None:
        print(f"[{now()}] {self.prefix}{redact(message)}", flush=True)


def request_stop(signum: int, _frame: Any) -> None:
    global RUNNING
    RUNNING = False
    print(f"[{now()}] received signal {signum}, shutting down", flush=True)
    if ACTIVE_GPU_PROCESS and ACTIVE_GPU_PROCESS.poll() is None:
        ACTIVE_GPU_PROCESS.terminate()


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def sleep_while_running(seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while RUNNING and time.monotonic() < deadline:
        time.sleep(min(0.25, deadline - time.monotonic()))


def parse_cookie_header(cookie_file: Path) -> str:
    raw = cookie_file.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"cookie file is empty: {cookie_file}")

    if raw.lower().startswith("cookie:"):
        raw = raw.split(":", 1)[1].strip()

    pairs: list[str] = []
    for item in raw.replace("\n", ";").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or name.lower() in COOKIE_ATTR_NAMES:
            continue
        pairs.append(f"{name}={value}")

    if not pairs:
        raise RuntimeError(f"no usable cookie pairs found in {cookie_file}")
    return "; ".join(pairs)


class Rpow2Api:
    def __init__(self, api_url: str, cookie_header: str, timeout: float) -> None:
        self.api_url = api_url.rstrip("/") + "/"
        self.cookie_header = cookie_header
        self.timeout = timeout

    def request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None
        headers = {
            "accept": "application/json",
            "cookie": self.cookie_header,
            "origin": "https://rpow2.com",
            "referer": "https://rpow2.com/",
            "user-agent": "rpow2-gpu-miner/1.0",
        }
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["content-type"] = "application/json"

        request = Request(urljoin(self.api_url, path.lstrip("/")), data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                if response.status == 204 or not body:
                    return None
                return json.loads(body.decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"request failed {method} {path}: {exc}") from exc

    def me(self) -> Any:
        return self.request("GET", "/me")

    def ledger(self) -> Any:
        return self.request("GET", "/ledger")

    def challenge(self) -> Any:
        return self.request("POST", "/challenge")

    def mint(self, challenge_id: str, solution_nonce: str) -> Any:
        return self.request(
            "POST",
            "/mint",
            {
                "challenge_id": challenge_id,
                "solution_nonce": str(solution_nonce),
            },
        )


def call_with_retry(label: str, func: Callable[[], Any], retry_sleep: float, logger: LiveLogger) -> Any:
    while RUNNING:
        try:
            return func()
        except Exception as exc:
            logger.log(f"{label} failed: {exc}; retrying in {retry_sleep:.1f}s")
            sleep_while_running(retry_sleep)
    return None


def run_cuda_search(args: argparse.Namespace, nonce_prefix: str, difficulty_bits: int, logger: LiveLogger) -> dict[str, Any] | None:
    global ACTIVE_GPU_PROCESS

    helper = Path(args.cuda_helper)
    if not helper.exists():
        raise RuntimeError(f"CUDA helper missing: {helper}")

    command = [
        str(helper),
        nonce_prefix,
        str(difficulty_bits),
        "--device",
        str(args.device),
        "--threads",
        str(args.cuda_threads),
        "--batch-iters",
        str(args.batch_iters),
        "--progress-ms",
        str(int(args.progress_interval * 1000)),
    ]
    if args.blocks:
        command.extend(["--blocks", str(args.blocks)])

    logger.log("gpu search starting " + " ".join(command))
    ACTIVE_GPU_PROCESS = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    result: dict[str, Any] | None = None
    try:
        assert ACTIVE_GPU_PROCESS.stdout is not None
        for raw_line in ACTIVE_GPU_PROCESS.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
                logger.log(
                    f"gpu found nonce={result['nonce']} hash={result['hash']} "
                    f"hashes~={result['hashes']} elapsed={float(result['elapsed']):.2f}s "
                    f"rate={float(result['rate_mh']):.2f} MH/s"
                )
                continue
            logger.log("gpu " + line)

        return_code = ACTIVE_GPU_PROCESS.wait()
        if return_code != 0 and RUNNING:
            raise RuntimeError(f"CUDA helper exited with code {return_code}")
        return result
    finally:
        if ACTIVE_GPU_PROCESS and ACTIVE_GPU_PROCESS.poll() is None:
            ACTIVE_GPU_PROCESS.terminate()
            try:
                ACTIVE_GPU_PROCESS.wait(timeout=2)
            except subprocess.TimeoutExpired:
                ACTIVE_GPU_PROCESS.kill()
        ACTIVE_GPU_PROCESS = None


class PipelineState:
    def __init__(self) -> None:
        self.minted = 0
        self.solved = 0
        self.lock = threading.Lock()

    def add_solved(self) -> int:
        with self.lock:
            self.solved += 1
            return self.solved

    def add_minted(self) -> int:
        with self.lock:
            self.minted += 1
            return self.minted

    def should_stop(self, args: argparse.Namespace) -> bool:
        with self.lock:
            return bool(args.once and self.minted >= 1) or bool(args.max_mints and self.minted >= args.max_mints)


def challenge_fetcher(
    worker_id: int,
    api: Rpow2Api,
    args: argparse.Namespace,
    challenge_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
    logger: LiveLogger,
) -> None:
    while RUNNING and not stop_event.is_set():
        try:
            challenge = api.challenge()
            challenge_queue.put(challenge, timeout=1)
            if args.fetch_log_interval and worker_id == 1:
                logger.log(f"prefetch queue={challenge_queue.qsize()}/{args.prefetch}")
        except queue.Full:
            continue
        except Exception as exc:
            logger.log(f"challenge worker {worker_id} failed: {exc}; retrying in {args.retry_sleep:.1f}s")
            sleep_while_running(args.retry_sleep)


def mint_worker(
    worker_id: int,
    api: Rpow2Api,
    args: argparse.Namespace,
    mint_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
    state: PipelineState,
    logger: LiveLogger,
) -> None:
    while RUNNING and not stop_event.is_set():
        try:
            item = mint_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            minted_token = call_with_retry(
                f"mint worker {worker_id}",
                lambda: api.mint(str(item["challenge_id"]), str(item["nonce"])),
                args.retry_sleep,
                logger,
            )
            if minted_token is None:
                continue
            minted = state.add_minted()
            logger.log(
                f"minted count={minted} solved={state.solved} "
                f"response={json.dumps(minted_token, ensure_ascii=False)}"
            )
            if state.should_stop(args):
                stop_event.set()
        finally:
            mint_queue.task_done()


def command_run(args: argparse.Namespace) -> int:
    install_signal_handlers()
    logger = LiveLogger()
    cookie = parse_cookie_header(args.cookie_file)
    api = Rpow2Api(args.api_url, cookie, args.timeout)
    state = PipelineState()
    stop_event = threading.Event()
    challenge_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, int(args.prefetch)))
    mint_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, int(args.mint_backlog)))

    logger.log(
        f"gpu protocol miner starting api={args.api_url} device={args.device} "
        f"cookie_pairs={len(cookie.split('; '))} "
        f"prefetch={args.prefetch} challenge_workers={args.challenge_workers} "
        f"mint_workers={args.mint_workers}"
    )

    fetch_threads = [
        threading.Thread(
            target=challenge_fetcher,
            args=(worker_id, api, args, challenge_queue, stop_event, logger),
            daemon=True,
        )
        for worker_id in range(1, max(1, int(args.challenge_workers)) + 1)
    ]
    mint_threads = [
        threading.Thread(
            target=mint_worker,
            args=(worker_id, api, args, mint_queue, stop_event, state, logger),
            daemon=True,
        )
        for worker_id in range(1, max(1, int(args.mint_workers)) + 1)
    ]

    for thread in fetch_threads + mint_threads:
        thread.start()

    try:
        while RUNNING and not stop_event.is_set():
            try:
                challenge = challenge_queue.get(timeout=1)
            except queue.Empty:
                continue

            challenge_id = challenge["challenge_id"]
            nonce_prefix = challenge["nonce_prefix"]
            difficulty_bits = int(challenge["difficulty_bits"])
            logger.log(
                f"challenge id={challenge_id} prefix={nonce_prefix} "
                f"difficulty_bits={difficulty_bits} device={args.device} "
                f"queue={challenge_queue.qsize()}/{args.prefetch}"
            )

            result = run_cuda_search(args, nonce_prefix, difficulty_bits, logger)
            challenge_queue.task_done()
            if not RUNNING or result is None:
                break

            solved = state.add_solved()
            mint_queue.put(
                {
                    "challenge_id": challenge_id,
                    "nonce": str(result["nonce"]),
                    "solved": solved,
                }
            )
            logger.log(
                f"solution queued solved={solved} mint_backlog={mint_queue.qsize()}/{args.mint_backlog}"
            )

            if args.once and solved >= 1:
                logger.log("once mode solved one challenge; waiting for mint")
                while RUNNING and not stop_event.is_set() and not state.should_stop(args):
                    time.sleep(0.25)
                break
    finally:
        stop_event.set()

    logger.log("gpu protocol miner stopped")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    cookie = parse_cookie_header(args.cookie_file)
    api = Rpow2Api(args.api_url, cookie, args.timeout)
    helper = Path(args.cuda_helper)
    me = api.me()
    ledger = api.ledger()
    print(f"me={json.dumps(me, ensure_ascii=False)}")
    print(f"ledger={json.dumps(ledger, ensure_ascii=False)}")
    print(f"cuda_helper={helper} exists={helper.exists()}")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    logger = LiveLogger()
    result = run_cuda_search(args, "00", int(args.difficulty_bits), logger)
    if result:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    return 1


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGTERM)

    for _ in range(40):
        if not is_process_running(pid):
            return
        time.sleep(0.25)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGKILL)


def read_last_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    data = path.read_bytes()[-131072:]
    return data.decode("utf-8", errors="replace").splitlines()[-limit:]


def daemon_start(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    log_file = Path(args.log_file)
    existing_pid = read_pid(pid_file)
    if is_process_running(existing_pid):
        print(f"already running pid={existing_pid}")
        print(f"log={log_file}")
        return 0

    if pid_file.exists():
        pid_file.unlink()

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "run",
        "--api-url",
        args.api_url,
        "--cookie-file",
        str(args.cookie_file),
        "--cuda-helper",
        str(args.cuda_helper),
        "--timeout",
        str(args.timeout),
        "--device",
        str(args.device),
        "--cuda-threads",
        str(args.cuda_threads),
        "--batch-iters",
        str(args.batch_iters),
        "--progress-interval",
        str(args.progress_interval),
        "--retry-sleep",
        str(args.retry_sleep),
        "--prefetch",
        str(args.prefetch),
        "--challenge-workers",
        str(args.challenge_workers),
        "--mint-workers",
        str(args.mint_workers),
        "--mint-backlog",
        str(args.mint_backlog),
        "--fetch-log-interval",
        str(args.fetch_log_interval),
    ]
    if args.blocks:
        command.extend(["--blocks", str(args.blocks)])
    if args.once:
        command.append("--once")
    if args.max_mints:
        command.extend(["--max-mints", str(args.max_mints)])

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab", buffering=0) as log_fp:
        log_fp.write(f"\n[{now()}] ===== gpu daemon start device={args.device} =====\n".encode())
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"started pid={proc.pid}")
    print(f"device={args.device}")
    print(f"log={log_file}")
    return 0


def daemon_stop(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    pid = read_pid(pid_file)
    if not is_process_running(pid):
        print("not running")
        if pid_file.exists():
            pid_file.unlink()
        return 0

    assert pid is not None
    kill_process_group(pid)
    if pid_file.exists():
        pid_file.unlink()
    print(f"stopped pid={pid}")
    return 0


def daemon_status(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    log_file = Path(args.log_file)
    pid = read_pid(pid_file)
    if is_process_running(pid):
        print(f"running pid={pid}")
    else:
        print("not running")
        if pid_file.exists():
            pid_file.unlink()

    print(f"pid_file={pid_file}")
    print(f"log={log_file}")
    lines = read_last_lines(log_file, args.lines)
    if lines:
        print("--- recent log ---")
        for line in lines:
            print(redact(line))
    return 0


def daemon_restart(args: argparse.Namespace) -> int:
    daemon_stop(args)
    return daemon_start(args)


def add_common_options(parser: argparse.ArgumentParser, include_lines: bool = False) -> None:
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--cookie-file", type=Path, default=DEFAULT_COOKIE_FILE)
    parser.add_argument("--cuda-helper", type=Path, default=DEFAULT_CUDA_HELPER)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--cuda-threads", type=int, default=256)
    parser.add_argument("--batch-iters", type=int, default=256)
    parser.add_argument("--progress-interval", type=float, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--challenge-workers", type=int, default=4)
    parser.add_argument("--mint-workers", type=int, default=4)
    parser.add_argument("--mint-backlog", type=int, default=1024)
    parser.add_argument("--fetch-log-interval", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-mints", type=int, default=0)
    if include_lines:
        parser.add_argument("--lines", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RPOW2 mining through direct API protocol and CUDA")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="start in background")
    add_common_options(start_parser)

    run_parser = subparsers.add_parser("run", help="run in foreground")
    add_common_options(run_parser)

    stop_parser = subparsers.add_parser("stop", help="stop background miner")
    add_common_options(stop_parser)

    restart_parser = subparsers.add_parser("restart", help="restart background miner")
    add_common_options(restart_parser)

    status_parser = subparsers.add_parser("status", help="show status and recent log")
    add_common_options(status_parser, include_lines=True)

    preflight_parser = subparsers.add_parser("preflight", help="check cookie, API access, and CUDA helper")
    add_common_options(preflight_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help="measure GPU speed with synthetic difficulty")
    add_common_options(benchmark_parser)
    benchmark_parser.add_argument("--difficulty-bits", type=int, default=28)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        raw_argv = ["start"]

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    try:
        if args.command == "run":
            return command_run(args)
        if args.command == "start":
            return daemon_start(args)
        if args.command == "stop":
            return daemon_stop(args)
        if args.command == "restart":
            return daemon_restart(args)
        if args.command == "status":
            return daemon_status(args)
        if args.command == "preflight":
            return command_preflight(args)
        if args.command == "benchmark":
            return command_benchmark(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(redact(f"error: {exc}"), file=sys.stderr)
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
