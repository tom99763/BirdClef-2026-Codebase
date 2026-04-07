"""
Launch training processes fully detached from the current shell session.
Uses os.setsid() to create a new session, making processes survive
when the parent bash shell exits (including in WSL2).
"""
import os
import subprocess
import sys
import json
import time

BASE = "/home/lab/BirdClef-2026-Codebase"

def is_finished(result_json):
    try:
        d = json.load(open(result_json))
        return bool(d.get("finished"))
    except:
        return False

def count_main_procs(pattern):
    """Count main training processes (not DataLoader workers)."""
    try:
        all_pids = subprocess.check_output(
            ["pgrep", "-f", pattern], text=True
        ).split()
    except subprocess.CalledProcessError:
        return 0, []

    pid_set = set(int(p) for p in all_pids)
    mains = []
    for pid in pid_set:
        try:
            status = open(f"/proc/{pid}/status").read()
            ppid = int([l for l in status.splitlines() if l.startswith("PPid:")][0].split()[1])
            if ppid not in pid_set:
                mains.append(pid)
        except:
            pass
    return len(mains), sorted(mains)

def launch(cmd, log_path):
    """Launch a command fully detached (new session, stdout/stderr → log)."""
    log = open(log_path, "a")
    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        env={**os.environ},
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # equivalent to setsid — detaches from parent session
        close_fds=True,
    )
    log.close()
    return proc.pid

def main():
    jobs = [
        {
            "name": "sed-b0-v15-no-sec",
            "result": f"{BASE}/outputs/sed-b0-v15-no-sec/result.json",
            "pattern": "train_sed.py.*sed_b0_v15",
            "cmd": [
                "python3", "train_sed.py",
                "--config", "configs/sed_b0_v15_no_sec.yaml",
                "--gpu", "0",
                "--pretrained_backbone", "checkpoints/embed-distill-b0-v1/best_backbone.pt",
            ],
            "env_gpu": "0",
            "log": f"{BASE}/outputs/v15_restart.log",
        },
        {
            "name": "embed-distill-b0-v4",
            "result": f"{BASE}/outputs/embed-distill-b0-v4/result.json",
            "pattern": "train_embed_distill.*b0_v4",
            "cmd": [
                "python3", "train_embed_distill.py",
                "--config", "configs/embed_distill_b0_v4.yaml",
                "--gpu", "1",
            ],
            "env_gpu": "1",
            "log": f"{BASE}/outputs/v4_distill_restart.log",
        },
    ]

    for job in jobs:
        name = job["name"]

        if is_finished(job["result"]):
            print(f"  [{name}] already finished — skip")
            continue

        n, pids = count_main_procs(job["pattern"])
        if n >= 1:
            print(f"  [{name}] running OK  (main PID={pids[0]})")
            if n > 1:
                print(f"  [{name}] WARNING: {n} copies — killing extras")
                for pid in pids[1:]:
                    try:
                        os.kill(pid, 9)
                    except:
                        pass
        else:
            # Set CUDA_VISIBLE_DEVICES in environment
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": job["env_gpu"]}
            log = open(job["log"], "a")
            proc = subprocess.Popen(
                job["cmd"],
                cwd=BASE,
                env=env,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            log.close()
            print(f"  [{name}] started  PID={proc.pid}")
            time.sleep(1)

if __name__ == "__main__":
    os.chdir(BASE)
    print(f"[launch_detached] Running from {BASE}")
    main()
    print("[launch_detached] Done")
