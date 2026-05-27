# main.py
"""Launch sim_interface child processes (robot_entry, optional viewer).

CLI flags (also usable from G1_deploy/start_sim_*.sh):
  --iface IFACE     -> forwarded as SIM_IFACE to children
  --no-viewer       -> skip viewer_entry.py (useful in headless reproductions)
  --auto-press SPEC -> forwarded as SIM_AUTO_PRESS to robot_entry.py
  --log-dir DIR     -> output dir; default sim_logs/<timestamp>
  --no-log          -> do not capture child stdout/stderr to files
"""
import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

subprocesses = []  # list[(name, Popen, log_fp)]


def cleanup_subprocesses():
    print("[MAIN] cleaning up child processes...")
    for entry in subprocesses:
        script_name, p = entry[0], entry[1]
        if p.poll() is None:
            print(f"[MAIN] terminating '{script_name}' pid={p.pid}")
            p.terminate()

    time.sleep(1)

    for entry in subprocesses:
        script_name, p = entry[0], entry[1]
        if p.poll() is None:
            print(f"[MAIN] forcing kill on '{script_name}' pid={p.pid}")
            p.kill()
    for entry in subprocesses:
        log_fp = entry[2] if len(entry) > 2 else None
        if log_fp is not None:
            try:
                log_fp.flush()
                log_fp.close()
            except Exception:
                pass
    print("[MAIN] cleanup done.")


def signal_handler(sig, frame):
    print(f"[MAIN] received signal {signal.Signals(sig).name}, exiting.")
    sys.exit(0)


def parse_args():
    p = argparse.ArgumentParser(description="sim_interface launcher")
    p.add_argument("--iface", default=os.getenv("SIM_IFACE", ""),
                   help="DDS network interface (forwarded as SIM_IFACE)")
    p.add_argument("--no-viewer", action="store_true",
                   help="Do not launch viewer_entry.py (headless mode)")
    p.add_argument("--auto-press", default=os.getenv("SIM_AUTO_PRESS", ""),
                   help='Auto button sequence, e.g. "1000:R2,3000:A,5000:X,7000:SimStart"')
    p.add_argument("--log-dir", default=os.getenv("SIM_LOG_DIR", ""),
                   help="Directory to dump child stdout/stderr and qpos trace. "
                        "Default: ./sim_logs/<timestamp>")
    p.add_argument("--no-log", action="store_true",
                   help="Disable auto file logging (children inherit terminal stdout)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    atexit.register(cleanup_subprocesses)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    base_env = os.environ.copy()
    if args.iface:
        base_env["SIM_IFACE"] = args.iface
    if args.auto_press:
        base_env["SIM_AUTO_PRESS"] = args.auto_press

    log_dir = None
    if not args.no_log:
        if args.log_dir:
            log_dir = Path(args.log_dir).resolve()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(__file__).resolve().parent / "sim_logs" / ts
        log_dir.mkdir(parents=True, exist_ok=True)
        base_env.setdefault("SIM_QPOS_TRACE_PATH", str(log_dir / "mujoco_qpos_trace.csv"))
        base_env.setdefault("SIM_LOG_DIR", str(log_dir))
        print(f"[MAIN] log_dir={log_dir}")

    scripts_to_run = ["robot_entry.py"]
    if not args.no_viewer:
        scripts_to_run.append("viewer_entry.py")

    print(f"[MAIN] launcher started, pid={os.getpid()}")
    if args.iface:
        print(f"[MAIN] SIM_IFACE={args.iface}")
    if args.auto_press:
        print(f"[MAIN] SIM_AUTO_PRESS={args.auto_press}")
    if args.no_viewer:
        print("[MAIN] viewer disabled (--no-viewer)")

    try:
        for script in scripts_to_run:
            log_fp = None
            stdout_target = None
            stderr_target = None
            if log_dir is not None:
                log_path = log_dir / f"{Path(script).stem}.log"
                log_fp = open(log_path, "w", buffering=1)
                stdout_target = log_fp
                stderr_target = subprocess.STDOUT
                print(f"[MAIN] {script} -> {log_path}")
            process = subprocess.Popen(
                [sys.executable, "-u", script],
                env=base_env,
                stdout=stdout_target,
                stderr=stderr_target,
            )
            subprocesses.append((script, process, log_fp))
            print(f"[MAIN] started '{script}' pid={process.pid}")

        print("[MAIN] all children launched. Ctrl+C to stop.")

        while True:
            for entry in subprocesses:
                script_name, p = entry[0], entry[1]
                return_code = p.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"child '{script_name}' exited unexpectedly, code={return_code}"
                    )
            time.sleep(1)

    except Exception as e:
        print(f"[MAIN] error: {e}")
    finally:
        print("[MAIN] launcher exiting.")
