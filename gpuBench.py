'''
GPU version against the CPU version: statistics of the same net on the same games, and the speed of one ES generation (800 children x 50 games).
Run in ~/venv-gpu after `python exportForGpu.py <net> <npz>` in tensorNetwork (training venv):

    python gpuBench.py net.npz [--children 800] [--games 50] [--rounds 3]

PAUSE_PIDS=pid,pid,... : a running CPU job is frozen (SIGSTOP) only around the timed rounds and continued (SIGCONT) right after, even on errors or Ctrl-C;
the CPU load is measured during the pause and printed.
'''
import argparse
import atexit
import os
import signal
import sys
import time

import numpy as np
from numba import cuda

import gpuEnv as ge

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--children", type=int, default=800)
ap.add_argument("--games", type=int, default=50)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--sigma", type=float, default=0.03)
a = ap.parse_args()
data = np.load(a.npz)
weights = [data[f"{k}{L}"] for L in range(4) for k in ("W", "B")]    # W0, B0, W1, B1, ...
pids = [int(p) for p in os.environ.get("PAUSE_PIDS", "").split(",") if p]
paused = []


def resume(*_):
    while paused:
        try:
            os.kill(paused.pop(), signal.SIGCONT)
        except ProcessLookupError:
            pass


def pause():
    for p in pids:
        os.kill(p, signal.SIGSTOP)
        paused.append(p)


atexit.register(resume)
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: (resume(), sys.exit(1)))


def cpuBusy(seconds=2.0):
    def read():
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(v), v[3] + v[4]    # total, idle + iowait
    t0, i0 = read(); time.sleep(seconds); t1, i1 = read()
    return 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))


# 1. the same net on the same 200 games (levels are identical, the forward pass differs in float32 rounding)
gb = ge.GpuBatch(children=1, games=200)
t = time.time()
r = gb.evaluate(weights, list(range(200)))
print(f"GPU, 1 child x 200 games (incl. compile): {time.time() - t:.1f} s")
frames = r["decisions"] * 5
ref = {k[4:]: float(data[k]) for k in data.files if k.startswith("ref_")}
gpu = dict(score=r["score"].mean(), shots=r["shots"].mean(), hits=r["hits"].mean(), frames=frames.mean(), steerChange=r["steerChange"].sum() / r["decisions"].sum(),
           detour=1000 * r["detour"].sum() / frames.sum(), deviation=1000 * r["deviation"].sum() / frames.sum())
print(f"{'':14s} {'CPU':>12s} {'GPU':>12s}")
for k in gpu:
    print(f"{k:14s} {ref[k]:12,.3f} {gpu[k]:12,.3f}")
print(f"(standard error of the CPU mean score: {ref['scoreSd']:,.0f})")

# 2. speed of one generation: parent + sigma * noise
rng = np.random.default_rng(1)
C = a.children
children = [np.repeat(w, C, axis=0) + a.sigma * rng.standard_normal((C,) + w.shape[1:]).astype(np.float32) for w in weights]
gb = ge.GpuBatch(children=C, games=a.games)
seeds = list(range(a.games))
t = time.time()
gb.evaluate(children, seeds)    # warm-up (compile for this shape, memory)
print(f"warm-up generation (children {C} x games {a.games}): {time.time() - t:.1f} s")
if pids:
    pause()
    time.sleep(3)
    print(f"CPU busy during the pause: {cpuBusy():.1f}%")
times = []
for i in range(a.rounds):
    cuda.synchronize()
    t = time.perf_counter()
    r = gb.evaluate(children, seeds)
    cuda.synchronize()
    times.append(time.perf_counter() - t)
    print(f"  round {i + 1}: {times[-1]:.2f} s  ({r['chunks']} kernel launches, mean score {r['score'].mean():,.0f}, longest game {int(r['decisions'].max() * 5):,} frames)")
resume()
print(f"GPU generation time: {np.mean(times):.2f} s (min {min(times):.2f}) -> {3600 / np.mean(times):,.0f} generations per hour")
