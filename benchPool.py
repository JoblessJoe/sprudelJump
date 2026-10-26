'''
One ES generation on a worker pool, with the children played one per task (group 1) or in groups of k per task (FastBatch.runGroup).
Unlike benchFastEnv.py (one core) this shows the END-OF-GENERATION TAIL: bigger tasks leave workers idle for longer at the end.
Needs numpy, numba, scipy only.

    python benchPool.py                 workers = physical cores - 2, games per child 50, ~26 children per worker (like 800 children on 31 workers)
    python benchPool.py --workers 6 --games 50 --rounds 4

Every group size plays the SAME children on the SAME seeds, so the scores must be identical; the generation time is what differs.
'''
import os
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import sys, time, argparse, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchFastEnv as bf
import fastEnv as fe


def work(task):
    kids, seeds = task
    if len(kids) == 1:
        return [bf.play(fe, kids[0], seeds)[0].tolist()]
    return bf.playGroup(fe, kids, seeds)[0].tolist()


def generation(pool, kids, seeds, k):
    tasks = [(kids[i:i + k], seeds) for i in range(0, len(kids), k)]
    t = time.perf_counter()
    results = []
    for r in pool.imap(work, tasks):    # in order, one task at a time handed out, like evaluatePopulation
        results.extend(r)
    return time.perf_counter() - t, results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2 - 2))
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--perworker", type=int, default=26, help="children per worker (800 children on 31 workers = 26)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--groups", default="1,2,3,4")
    a = ap.parse_args()
    ks = [int(x) for x in a.groups.split(",")]
    net = bf.loadNet()
    children = a.workers * a.perworker
    kids = bf.makeChildren(net, children, seed=5)
    print(f"{a.workers} workers, {children} children x {a.games} games per generation, groups {ks}, {a.rounds} rounds (order rotated)\n")
    with mp.Pool(a.workers) as pool:
        seeds = [int(s) for s in np.random.RandomState(0).randint(0, 2 ** 31, a.games)]
        for k in ks:    # compile / load the numba cache in every worker
            generation(pool, kids[:a.workers * max(ks)], seeds[:10], k)
        times = {k: [] for k in ks}
        reference, same = None, True
        for r in range(a.rounds):
            seeds = [int(s) for s in np.random.RandomState(100 + r).randint(0, 2 ** 31, a.games)]
            order = ks[r % len(ks):] + ks[:r % len(ks)]
            ref = None
            for k in order:
                dt, res = generation(pool, kids, seeds, k)
                times[k].append(dt)
                if ref is None:
                    ref = res
                elif res != ref:
                    same = False
                    print(f"  !! scores differ (group {k}, round {r})")
            print(f"  round {r}: " + "  ".join(f"group {k}: {times[k][-1]:6.2f} s" for k in ks), flush=True)
    base = np.mean(times[ks[0]])
    print("\ngroup size   generation time (mean / min)   vs group " + str(ks[0]))
    for k in ks:
        m = np.mean(times[k])
        print(f"{k:10d}   {m:7.2f} s / {np.min(times[k]):6.2f} s          {(base / m - 1) * 100:+6.1f}% faster (by min {(np.min(times[ks[0]]) / np.min(times[k]) - 1) * 100:+.1f}%)")
    print("\nscores identical for every group size:", "YES" if same else "NO  <-- wrong")
