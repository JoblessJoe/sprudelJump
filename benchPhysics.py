'''
Where does one physics frame (fastEnv.stepGame) spend its time? Timing-only experiment: for each section of stepGame a variant is compiled with that
section removed (the game it plays is then WRONG, only the time matters), and all variants are timed on the same frozen mid-game states.
A section's share = (full - variant) / full.  Needs numpy, numba, scipy only; run on an otherwise idle machine, one process:

    python benchPhysics.py            (about 1-2 minutes, mostly compiling)
'''
import os
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import sys, re, time, inspect
import numpy as np
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchFastEnv as bf
import fastEnv as fe

REPEAT = bf.REPEAT


def cut(src, start, end, stub):
    '''Replaces src[start:end) (anchors are literal text, `end` is kept) with stub.'''
    i = src.index(start)
    j = src.index(end, i + len(start))
    return src[:i] + stub + src[j:]


def variants():
    src = inspect.getsource(fe.stepGame.py_func)
    many = inspect.getsource(fe.stepMany.py_func)
    out = {}
    out["full (control)"] = src
    out["no bounce loop"] = cut(src, "    candidate = False\n    if velY > 0:", "    nM = I[g, NM]\n    fatal = False", "    removed = False\n")
    out["no monster loop"] = cut(src, "    nM = I[g, NM]\n    fatal = False\n    total = F[g, TOTAL]\n    k = 0\n", "    # bullets", "    nM = I[g, NM]\n    fatal = False\n    total = F[g, TOTAL]\n")
    out["no bullets (shoot, move, hit)"] = cut(src, "    # bullets\n", "    # scrolling.", "    nB = I[g, NB]\n    bulletGone = 0\n    monsterHit = 0\n")
    out["  no bullet-monster collision loop"] = cut(src, "    bulletGone = 0\n    monsterHit = 0\n    for i in range(nB):\n        bx = BU[g, i, 0]", "    if bulletGone != 0:", "    bulletGone = 0\n    monsterHit = 0\n")
    out["  no bullet shift while scrolling"] = src.replace("        for i in range(nB):\n            BU[g, i, 1] += scroll\n", "")
    out["  no monster shift while scrolling"] = src.replace("        for i in range(nM):\n            MO[g, i, 1] += scroll\n", "")
    out["  no bottom filter for bullets"] = cut(src, "    k = 0\n    for i in range(nB):\n        if BU[g, i, 1] <= SCREEN_HEIGHT:", "    I[g, NP] = nP", "    nM = nM\n    nB = nB\n")
    out["no scroll pass"] = src.replace("    scrolled = playerY < SCROLL_THRESHOLD_Y\n", "    scrolled = False\n")
    out["no spawn loop"] = cut(src, "    while I[g, NP] > 0:\n        if not minY > 0:", "    F[g, TOPY] = minY", "")
    out["no bookkeeping tail"] = cut(src, "    altitude = F[g, TOTAL]", "    return fatal or fell or stuck", "    fell = False\n    stuck = False\n    I[g, CAUSE] = 0\n")
    return out, many


def compile_variant(name, stepSrc, manySrc):
    ns = dict(fe.__dict__)
    s = stepSrc.replace("def stepGame(", "def stepGameV(").replace("@njit(cache=True)\n", "")
    m = manySrc.replace("def stepMany(", "def stepManyV(").replace("stepGame(", "stepGameV(").replace("@njit(cache=True)\n", "")
    code = "from numba import njit\n@njit\n" + s + "\n@njit\n" + m
    exec(compile(code, f"<variant {name}>", "exec"), ns)
    return ns["stepManyV"]


def frozenStates(net, decisions=(300, 1200, 2500, 4000), games=200):
    wTs, bs = net
    seeds = [int(s) for s in np.random.RandomState(7).randint(0, 2 ** 31, games)]
    b = fe.FastBatch(seeds, None, 0, 0.0, 0.0, 1.0, stable=bf.LAYOUT)
    alive, obs = b.startLoop(bf.N_IN)
    n = len(alive)
    applied = np.full(games, 0.5)
    states = []
    for d in range(max(decisions) + 1):
        if d in decisions and n:
            x = bf.forward(fe, obs[:n], wTs, bs)
            states.append(([a.copy() for a in (b.F, b.I, b.PL, b.MO, b.BU, b.MT)], alive[:n].copy(), x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)))
        if not n:
            break
        n = b.stepObserve(alive, n, bf.forward(fe, obs[:n], wTs, bs), applied, bf.SMOOTH, REPEAT, decide=False)
    return states, b


def main():
    net = bf.loadNet()
    bf.header()
    states, b = frozenStates(net)
    nAlive = sum(len(s[1]) for s in states)
    print(f"{len(states)} frozen states, {nAlive} game-decisions in total ({nAlive * REPEAT} game-frames per pass)\n")
    work = [[a.copy() for a in s[0]] for s in states]
    maxNP = b.maxNoProgress
    PASSES = 10    # decisions played from each frozen state before it is restored (restoring is expensive, so it stays out of the timing)

    def measure(stepMany):
        '''Seconds spent inside stepMany and game-frames played, over PASSES decisions from every frozen state; only games still alive are stepped.'''
        spent, frames = 0.0, 0
        for w, s in zip(work, states):
            for a, v in zip(w, s[0]):
                a[...] = v
            for _ in range(PASSES):
                m = w[1][s[1], fe.DONE] == 0
                al, st, sh = s[1][m], s[2][m], s[3][m]
                if len(al) == 0:
                    break
                t = time.perf_counter()
                stepMany(w[0], w[1], w[2], w[3], w[4], w[5], al, st, sh, REPEAT, maxNP)
                spent += time.perf_counter() - t
                frames += len(al) * REPEAT
        return spent, frames

    vs = variants()
    funcs = {}
    for name, srcText in vs[0].items():
        funcs[name] = compile_variant(name, srcText, vs[1])
        measure(funcs[name])    # compile + warm up
        print(f"  compiled: {name}", flush=True)
    # interleaved rounds (every variant once per round, order rotated), so that drift of the machine hits all variants alike; best round counts
    best = {name: (float("inf"), 1) for name in funcs}
    names = list(funcs)
    for rnd in range(20):
        for name in names[rnd % len(names):] + names[:rnd % len(names)]:
            r = measure(funcs[name])
            if r[0] / r[1] < best[name][0] / best[name][1]:
                best[name] = r
    results = {name: spent / frames * 1e9 for name, (spent, frames) in best.items()}
    full = results["full (control)"]
    print(f"\n{PASSES} decisions from each of {len(states)} frozen states, only games still alive are stepped.\n")
    print(f"{'variant (section removed)':36s} {'ns per game-frame':>18s} {'saved':>8s}")
    for name, t in results.items():
        print(f"{name:36s} {t:18.1f} {(full - t) / full * 100:7.1f}%")
    print("\nSections overlap a little (removing one changes what the others see), so the savings are indications, not an exact partition.")


if __name__ == "__main__":
    main()
