#!/usr/bin/env python
"""Run task files on the GPUs of ONE node, several tasks per GPU.

    python scripts/pool.py tasks/smoke/1_reference.txt tasks/smoke/2_unlearn.txt \
        --slots-per-gpu 6 --log-dir tasks/smoke/logs

* Task files run in the order given, with a barrier between files (all
  reference models exist before any unlearning run starts).
* ``slots-per-gpu`` worker processes are pinned to each visible GPU via
  ``CUDA_VISIBLE_DEVICES`` (8 GPUs x 7 slots = 56 concurrent runs).
* Each task's stdout/stderr goes to ``<log-dir>/<file>/<line>.log``; a status
  line (rc, seconds, gpu, command) is appended to ``<log-dir>/status.tsv``.
* Failed tasks are retried ``--retries`` times. Tasks skip themselves when
  their output exists, so re-running the pool (e.g. after the 24 h Slurm limit)
  resumes where it stopped.
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g != ""]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def run_file(path, gpus, slots, log_dir, retries, threads_per_task):
    tasks = [l.strip() for l in Path(path).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    flog = log_dir / Path(path).stem
    flog.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for i, t in enumerate(tasks):
        q.put((i, t, 0))
    lock = threading.Lock()
    failed, done = [], [0]
    start = time.time()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS=str(threads_per_task), MKL_NUM_THREADS=str(threads_per_task),
                   PYTHONUNBUFFERED="1")
        while True:
            try:
                i, task, attempt = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            with open(flog / f"{i:05d}.log", "a") as f:
                f.write(f"# attempt {attempt} gpu {gpu}\n# {task}\n")
                f.flush()
                rc = subprocess.call(task, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
            dt = time.time() - t0
            with lock:
                with open(log_dir / "status.tsv", "a") as s:
                    s.write(f"{Path(path).stem}\t{i}\t{rc}\t{dt:.1f}\t{gpu}\t{attempt}\t{task}\n")
                if rc != 0 and attempt < retries:
                    q.put((i, task, attempt + 1))
                    continue
                if rc != 0:
                    failed.append(i)
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(tasks):
                    el = time.time() - start
                    print(f"[{Path(path).name}] {done[0]}/{len(tasks)} done, {len(failed)} failed, "
                          f"{el / 60:.1f} min elapsed", flush=True)

    # round-robin over GPUs, so a short task list is still spread over all GPUs
    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for _ in range(slots) for g in gpus]
    for th in threads:
        th.start()
        time.sleep(0.5)  # stagger start-up (data loading)
    for th in threads:
        th.join()
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_files", nargs="+")
    p.add_argument("--slots-per-gpu", type=int, default=6)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--threads-per-task", type=int, default=1)
    a = p.parse_args()

    gpus = visible_gpus()
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPUs {gpus}, {a.slots_per_gpu} slots each -> {len(gpus) * a.slots_per_gpu} workers",
          flush=True)
    any_failed = False
    for f in a.task_files:
        failed = run_file(f, gpus, a.slots_per_gpu, log_dir, a.retries, a.threads_per_task)
        if failed:
            any_failed = True
            print(f"[{f}] FAILED tasks (line index): {failed}  -> see {log_dir}", flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
