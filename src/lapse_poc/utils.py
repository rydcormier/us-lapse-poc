from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence  # noqa: UP035

import numpy as np
import torch


@dataclass(frozen=True)
class RunResult:
    cmd: list[str]
    returncode: int
    output: str


def run_stream(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    unbuffered: bool = True,
    echo: bool = True,
    tail_on_error: int = 400,
) -> RunResult:
    """
    Run a subprocess and stream output live when possible.
    If streaming produces no output (common in some Windows/Jupyter failures),
    fall back to subprocess.run(capture_output=True) so you always get the error text.
    """
    cmd_list = [str(x) for x in cmd]

    run_env = os.environ.copy()
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})
    if unbuffered:
        run_env["PYTHONUNBUFFERED"] = "1"

    if echo:
        print("\n" + "=" * 120)
        print("CWD:", str(cwd) if cwd is not None else os.getcwd())
        print("CMD:", " ".join(cmd_list))
        print("=" * 120)

    # --- Attempt streaming mode ---
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(cwd) if cwd is not None else None,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if line == "" and proc.poll() is not None:
                break
            lines.append(line)
            print(line, end="")  # live output

        rc = proc.wait()
        output = "".join(lines)

    except Exception as e:
        # If streaming setup failed, fall back to capture mode.
        if echo:
            print("\n[run_stream] Streaming failed, falling back to capture mode. Reason:", repr(e))
        rc = 999
        output = ""

    # --- If streaming produced nothing or failed, fall back to capture_output ---
    if (rc != 0 and len(lines) == 0) or (rc == 999):
        completed = subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd is not None else None,
            env=run_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        rc = completed.returncode
        output = completed.stdout or ""

        # Print captured output so it's visible in notebook
        if output:
            print(output, end="" if output.endswith("\n") else "\n")

    if rc != 0 and check:
        # Always print tail on error (and full output is already printed above in capture mode)
        if tail_on_error > 0 and lines:
            tail = "".join(lines[-tail_on_error:])
            print("\n" + "!" * 120)
            print(f"Command failed (exit={rc}). Last {min(tail_on_error, len(lines))} lines:")
            print("!" * 120)
            print(tail)

        # Include output in the exception object
        raise subprocess.CalledProcessError(rc, cmd_list, output=output)

    if echo:
        print("\n" + "-" * 120)
        print(f"Done (exit={rc}).")
        print("-" * 120)

    return RunResult(cmd=cmd_list, returncode=rc, output=output)


def run_python_module_stream(
    module: str,
    args: Sequence[str] = (),
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    echo: bool = True,
) -> RunResult:
    """
    Convenience wrapper for: python -u -m <module> <args...>
    Uses the same interpreter as the current process (important in notebooks).
    """
    cmd = [sys.executable, "-u", "-m", module, *map(str, args)]
    return run_stream(cmd, cwd=cwd, env=env, check=check, echo=echo)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
