'''
Speed benchmark for fastEnv.py (needs only numpy, numba and scipy; no torch, no tensorNetwork checkout).

    python benchFastEnv.py profile            where the time goes (whole evaluation + parts on a frozen mid-game state), current fastEnv.py
    python benchFastEnv.py ab                 old fastEnv (git: main) against the current file, alternating rounds, scores must be identical
    python benchFastEnv.py ab --old FILE      the old version from a file instead of git
    python benchFastEnv.py group              k children one by one against k children in one lockstep loop (FastBatch.runGroup), scores must be identical

Plays the current best Sprudler net (bench/net_repeat5_36k.npz: 33-33-33-33-2, landing layout, action repeat 5, smoothing 0.96875)
exactly like proSprudler.evaluateNetworkFast does, in ONE process, so run it on an otherwise idle machine. Exact comparisons need
single-threaded BLAS (set below). The absolute numbers depend on the machine; only the old/new RATIO transfers between machines.
'''
import os
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import sys, time, argparse, importlib.util, subprocess, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NET = os.path.join(HERE, "bench", "net_repeat5_36k.npz")
LAYOUT, REPEAT, SMOOTH, N_IN = (3, 2, 1, 2, 0, 1), 5, 1 - 0.5 ** 5, 33
GAMES = 200
pc = time.perf_counter


def loadModule(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def loadNet():
    d = np.load(NET)
    n = len([k for k in d.files if k.startswith("w")])
    return tuple(d[f"w{i}"].T.copy() for i in range(n)), tuple(d[f"b{i}"] for i in range(n))


def forward(fe, x, wTs, bs):
    '''Network output like evaluateNetworkFast: compiled matmuls, numpy's own float32 sigmoid.'''
    if hasattr(fe, "forwardLogits"):
        x = fe.forwardLogits(x, wTs, bs)
    else:
        for L, (wT, b) in enumerate(zip(wTs, bs)):
            x = x @ wT + b
            if L < len(wTs) - 1:
                x = np.maximum(x, 0)
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


def play(fe, net, seeds, maxStart=None, minStart=0, zero=0.0, maxFrames=100_000, timed=False):
    '''One full evaluation. Returns (scores, seconds, forwardSeconds, decisions).'''
    wTs, bs = net
    b = fe.FastBatch(seeds, maxStart, minStart, zero, 0.0, 1.0, stable=LAYOUT)
    alive, obs = b.startLoop(N_IN)
    n = len(alive)
    applied = np.full(len(seeds), 0.5)
    frames = decisions = 0
    tFwd = 0.0
    t0 = pc()
    if hasattr(b, "run") and not timed:    # the current evaluation loop (see FastBatch.run); the old fastEnv has only the loop below
        n, frames = b.run(alive, n, wTs, bs, applied, SMOOTH, REPEAT, False, 0, maxFrames)
        return b.scores().copy(), pc() - t0, tFwd, frames // REPEAT
    while n and frames < maxFrames:
        if timed:
            t1 = pc()
        x = forward(fe, obs[:n], wTs, bs)
        if timed:
            tFwd += pc() - t1
        n = b.stepObserve(alive, n, x, applied, SMOOTH, REPEAT, decide=False)
        frames += REPEAT
        decisions += 1
    return b.scores().copy(), pc() - t0, tFwd, decisions


def gitOld():
    '''The old fastEnv.py from git (branch main), saved next to a private cache dir so numba caches both versions separately.'''
    os.makedirs(os.path.join(HERE, "bench", "_old"), exist_ok=True)
    path = os.path.join(HERE, "bench", "_old", "fastEnv_old.py")
    src = subprocess.run(["git", "show", "main:fastEnv.py"], cwd=HERE, capture_output=True, text=True, check=True).stdout
    with open(path, "w") as f:
        f.write(src)
    return path


def header():
    print(f"machine: {platform.processor() or platform.machine()}, python {platform.python_version()}, numpy {np.__version__}")
    import numba
    print(f"numba {numba.__version__}, single-threaded BLAS ({os.environ['OPENBLAS_NUM_THREADS']}), {GAMES} games per evaluation, "
          f"repeat {REPEAT}, landing layout {LAYOUT}\n")


def profile(fe, net, rounds):
    wTs, bs = net
    seeds = [int(s) for s in np.random.RandomState(1).randint(0, 2 ** 31, GAMES)]
    play(fe, net, seeds[:20])    # compile everything
    print("whole evaluation (start 0):")
    for r in range(rounds):
        sc, t, tf, dec = play(fe, net, [int(s) for s in np.random.RandomState(r).randint(0, 2 ** 31, GAMES)], timed=True)
        frames = dec * REPEAT
        print(f"  round {r}: {t:6.2f} s, mean score {sc.mean():8.0f}, {dec:6d} decisions (longest game), forward {tf / t * 100:4.1f}% of the time, "
              f"{t / max(1, sc.size) * 1e3:6.1f} ms per game")
    # frozen mid-game state: play DECISIONS decisions, then time the parts in isolation on copies of that state
    decisions = 2500
    b = fe.FastBatch(seeds, None, 0, 0.0, 0.0, 1.0, stable=LAYOUT)
    alive, obs = b.startLoop(N_IN)
    n = len(alive)
    applied = np.full(GAMES, 0.5)
    for _ in range(decisions):
        if not n:
            break
        n = b.stepObserve(alive, n, forward(fe, obs[:n], wTs, bs), applied, SMOOTH, REPEAT, decide=False)
    print(f"\nparts on a frozen state after {decisions} decisions ({decisions * REPEAT:,} frames), {n} of {GAMES} games still alive:")
    arrays = (b.F, b.I, b.PL, b.MO, b.BU, b.MT)
    saved = [a.copy() for a in arrays]
    al = alive[:n].copy()
    x = forward(fe, obs[:n], wTs, bs)
    steer, shoot = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    out32, out64 = np.zeros((GAMES, N_IN), np.float32), np.zeros((GAMES, N_IN))

    def restore():
        for a, s in zip(arrays, saved):
            a[...] = s

    def timeit(f, reps=300, withRestore=False):
        f()
        t = pc()
        for _ in range(reps):
            if withRestore:
                restore()
            f()
        return (pc() - t) / reps

    physics = lambda: fe.stepMany(b.F, b.I, b.PL, b.MO, b.BU, b.MT, al.copy(), steer, shoot, REPEAT, b.maxNoProgress)
    observe = lambda: fe.observeFloat32(b.F, b.I, b.PL, b.MO, al, n, out32, *b.loopMode)
    fused = lambda: fe.stepObserveMany(b.F, b.I, b.PL, b.MO, b.BU, b.MT, al.copy(), n, x, np.full(GAMES, 0.5), SMOOTH, REPEAT, b.maxNoProgress,
                                       False, False, out32, *b.loopMode)
    fwd = lambda: forward(fe, obs[:n], wTs, bs)
    base = timeit(restore, withRestore=False)
    tPhys = timeit(physics, withRestore=True) - base
    tObs = timeit(observe)
    tFused = timeit(fused, withRestore=True) - base
    tFwd = timeit(fwd)
    print(f"  {'part':34s} {'us per decision':>16s} {'ns per game':>12s} {'share of one decision':>22s}")
    total = tFwd + tFused
    for name, t in (("physics only (stepMany, 5 frames)", tPhys), ("observation only", tObs), ("fused step + observe (what runs)", tFused),
                    ("network forward + sigmoid", tFwd)):
        print(f"  {name:34s} {t * 1e6:16.1f} {t / n * 1e9:12.0f} {t / total * 100:21.1f}%")
    print(f"  physics per game-frame: {tPhys / n / REPEAT * 1e9:.0f} ns")
    # observation split: the same call without the landing prediction (23 inputs instead of 33), and the float32 zero + copy around it
    mode = list(b.loopMode)
    noLand = tuple(mode[:-1] + [False])
    out32n, out64n = np.zeros((GAMES, N_IN - 10), np.float32), np.zeros((GAMES, N_IN - 10))
    tObsNoLand = timeit(lambda: fe.observeFloat32(b.F, b.I, b.PL, b.MO, al, n, out32n, *noLand))
    tObsRaw = timeit(lambda: fe.observeMany(b.F, b.I, b.PL, b.MO, al, out64, *b.loopMode))
    print(f"  observation without landing prediction: {tObsNoLand * 1e6:.1f} us  -> landing prediction = {(tObs - tObsNoLand) * 1e6:.1f} us "
          f"({(tObs - tObsNoLand) / tObs * 100:.0f}% of the observation, {(tObs - tObsNoLand) / total * 100:.0f}% of a decision)")
    print(f"  observeMany alone (no zeroing / float32 copy): {tObsRaw * 1e6:.1f} us -> zero + copy = {(tObs - tObsRaw) * 1e6:.1f} us")


def overhead(fe, net):
    '''Cost of ONE decision (forward + sigmoid + stepObserve, as in play()) against the number of games still alive: the intercept is the fixed
    cost per decision (Python loop, numpy calls, dispatch), the slope the cost per game. Matters because the longest game of a batch keeps the
    loop running while few games are left.'''
    wTs, bs = net
    seeds = [int(s) for s in np.random.RandomState(3).randint(0, 2 ** 31, 200)]
    b = fe.FastBatch(seeds, None, 0, 0.0, 0.0, 1.0, stable=LAYOUT)
    alive, obs = b.startLoop(N_IN)
    n = len(alive)
    applied = np.full(200, 0.5)
    for _ in range(300):
        n = b.stepObserve(alive, n, forward(fe, obs[:n], wTs, bs), applied, SMOOTH, REPEAT, decide=False)
    arrays = (b.F, b.I, b.PL, b.MO, b.BU, b.MT)
    snap = [a.copy() for a in arrays]
    al0 = alive[:n].copy()
    print(f"\ncost of one decision against the games still alive (30 decisions from a frozen state at decision 300, best of 7):")
    print(f"  {'games alive':>11s} {'us per decision':>16s} {'ns per game':>12s}")
    xs, ys = [], []
    for m in (1, 2, 5, 10, 25, 50, 100, 200):
        if m > len(al0):
            continue
        best = float("inf")
        for rep in range(7):
            for a, v in zip(arrays, snap):
                a[...] = v
            al = al0[:m].copy()
            ap_ = np.full(200, 0.5)
            fe.observeFloat32(b.F, b.I, b.PL, b.MO, al, m, b.obs, *b.loopMode)
            k, steps, gd = m, 0, 0
            t = pc()
            if hasattr(b, "run"):    # the current loop; games rarely end within 30 decisions of this state, so m is the number alive throughout
                b.run(al, k, wTs, bs, ap_, SMOOTH, REPEAT, False, 0, 30 * REPEAT)
                steps, gd = 30, 30 * m
            else:
                for _ in range(30):
                    if k == 0:
                        break
                    x = forward(fe, b.obs[:k], wTs, bs)
                    gd += k
                    k = b.stepObserve(al, k, x, ap_, SMOOTH, REPEAT, decide=False)
                    steps += 1
            el = pc() - t
            best = min(best, el / steps * 1e6)
            avgAlive = gd / steps
        xs.append(avgAlive); ys.append(best)
        print(f"  {avgAlive:11.1f} {best:16.1f} {best / avgAlive * 1e3:12.0f}")
    slope, icpt = np.polyfit(xs, ys, 1)
    print(f"  fit: {icpt:.1f} us fixed per decision + {slope * 1e3:.0f} ns per game")
    # the pieces of the fixed cost, with one game alive
    for a, v in zip(arrays, snap):
        a[...] = v
    al = al0[:1].copy(); ap_ = np.full(200, 0.5)
    fe.observeFloat32(b.F, b.I, b.PL, b.MO, al, 1, b.obs, *b.loopMode)
    x1 = forward(fe, b.obs[:1], wTs, bs)
    def tm(f, reps=3000):
        f(); t = pc()
        for _ in range(reps): f()
        return (pc() - t) / reps * 1e6
    tLog = tm(lambda: fe.forwardLogits(b.obs[:1], wTs, bs)) if hasattr(fe, "forwardLogits") else float("nan")
    xl = fe.forwardLogits(b.obs[:1], wTs, bs) if hasattr(fe, "forwardLogits") else None
    def sig():
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(-xl))
    tSig = tm(sig)
    tStep = tm(lambda: fe.stepObserveMany(b.F, b.I, b.PL, b.MO, b.BU, b.MT, al.copy(), 1, x1, ap_, SMOOTH, REPEAT, b.maxNoProgress, False, False, b.obs, *b.loopMode))
    print(f"  pieces with 1 game alive (old way): compiled forward {tLog:.1f} us, numpy sigmoid {tSig:.1f} us, stepObserveMany (the game itself, 1 game) {tStep:.1f} us")
    if hasattr(fe, "forwardNegated"):
        eNeg = fe.forwardNegated(b.obs[:1], wTs, bs)
        tNeg = tm(lambda: fe.forwardNegated(b.obs[:1], wTs, bs))
        def expInPlace():
            with np.errstate(over="ignore"):
                np.exp(eNeg.copy(), out=eNeg.copy())
        tExp = tm(expInPlace)
        tStepE = tm(lambda: fe.stepObserveMany(b.F, b.I, b.PL, b.MO, b.BU, b.MT, al.copy(), 1, np.exp(eNeg), ap_, SMOOTH, REPEAT, b.maxNoProgress, False, True, b.obs, *b.loopMode))
        print(f"  pieces with 1 game alive (new way): forwardNegated {tNeg:.1f} us, np.exp (+2 small copies) {tExp:.1f} us, stepObserveMany(fromExp) incl. np.exp {tStepE:.1f} us")


def makeChildren(net, k, sigma=0.05, seed=0):
    '''k ES-like children of the net: weights + sigma * noise (float32).'''
    wTs, bs = net
    rng = np.random.RandomState(seed)
    kids = []
    for _ in range(k):
        kids.append((tuple((w + rng.randn(*w.shape).astype(np.float32) * np.float32(sigma)).astype(np.float32) for w in wTs),
                     tuple((b + rng.randn(*b.shape).astype(np.float32) * np.float32(sigma)).astype(np.float32) for b in bs)))
    return kids


def stack(kids):
    nL = len(kids[0][0])
    return (tuple(np.ascontiguousarray(np.stack([kid[0][L] for kid in kids])) for L in range(nL)),
            tuple(np.ascontiguousarray(np.stack([kid[1][L] for kid in kids])) for L in range(nL)))


def playGroup(fe, kids, seeds, maxStart=None, minStart=0, zero=0.0, maxFrames=100_000):
    '''k children in ONE FastBatch / one loop (FastBatch.runGroup). Returns (scores [k, S], seconds).'''
    k, S = len(kids), len(seeds)
    ws, bs = stack(kids)
    t0 = pc()
    b = fe.FastBatch(list(seeds) * k, maxStart, minStart, zero, 0.0, 1.0, stable=LAYOUT)
    alive, obs = b.startLoop(N_IN)
    applied = np.full(k * S, 0.5)
    b.runGroup(alive, len(alive), ws, bs, S, applied, SMOOTH, REPEAT, False, 0, maxFrames)
    return b.scores().reshape(k, S).copy(), pc() - t0


def group(fe, net, rounds):
    cases = ((50, (2, 3, 4, 8)), (200, (2, 4)))
    for S, ks in cases:
        for k in ks:
            kids = makeChildren(net, k)
            playGroup(fe, kids[:2], list(range(S)))
            for kid in kids[:2]:
                play(fe, kid, list(range(S)))                                   # compile
            single, grouped, same = [], [], True
            for r in range(rounds):
                seeds = [int(x) for x in np.random.RandomState(200 + r).randint(0, 2 ** 31, S)]
                order = ("single", "group") if r % 2 == 0 else ("group", "single")
                for name in order:
                    if name == "single":
                        t, sc = 0.0, []
                        for kid in kids:
                            sco, dt, _, _ = play(fe, kid, seeds)
                            t += dt; sc.append(sco)
                        single.append(t); resS = np.array(sc)
                    else:
                        resG, dt = playGroup(fe, kids, seeds)
                        grouped.append(dt)
                if not np.array_equal(resS, resG):
                    same = False
                    print(f"  !! scores differ in round {r} (k={k}, {S} games)")
            sg, gp = np.array(single), np.array(grouped)
            print(f"{S:3d} games per child, group of {k}: one by one {sg.mean() / k * 1e3:7.1f} ms per child (min {sg.min() / k * 1e3:.1f})  "
                  f"grouped {gp.mean() / k * 1e3:7.1f} ms per child (min {gp.min() / k * 1e3:.1f})  -> {(sg.mean() / gp.mean() - 1) * 100:+5.1f}% faster "
                  f"(by min {(sg.min() / gp.min() - 1) * 100:+5.1f}%)  scores identical: {'YES' if same else 'NO'}")


def ab(new, old, net, rounds):
    cases = (("start 0", None, 0, 0.0), ("hard starts 20-30k + half from 0", 30000, 20000, 0.5))
    play(new, net, list(range(50)))
    play(old, net, list(range(50)))    # compile both
    allSame = True
    for label, mx, mn, z in cases:
        times = {"old": [], "new": []}
        for r in range(rounds):
            seeds = [int(s) for s in np.random.RandomState(100 + r).randint(0, 2 ** 31, GAMES)]
            result = {}
            for name in (("old", "new") if r % 2 == 0 else ("new", "old")):    # alternate who goes first
                sc, t, _, _ = play(new if name == "new" else old, net, seeds, mx, mn, z)
                times[name].append(t)
                result[name] = sc
            if not np.array_equal(result["old"], result["new"]):
                allSame = False
                print(f"  !! scores differ in round {r} ({label})")
        o, nw = np.array(times["old"]), np.array(times["new"])
        print(f"{label:34s} old {o.mean():6.2f} s (min {o.min():.2f})  new {nw.mean():6.2f} s (min {nw.min():.2f})  "
              f"-> {(o.mean() / nw.mean() - 1) * 100:+.1f}% faster (by min: {(o.min() / nw.min() - 1) * 100:+.1f}%)   noise: old +-{o.std() / o.mean() * 100:.1f}%, new +-{nw.std() / nw.mean() * 100:.1f}%")
    print("\nscores identical in every game:", "YES" if allSame else "NO  <-- not a valid optimization")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("profile", "ab", "group"))
    ap.add_argument("--old", help="old fastEnv.py (default: `git show main:fastEnv.py`)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--games", type=int, default=200, help="games per evaluation (training now uses 50 per child)")
    a = ap.parse_args()
    GAMES = a.games
    header()
    net = loadNet()
    new = loadModule(os.path.join(HERE, "fastEnv.py"), "fastEnv")
    if a.mode == "group":
        group(new, net, a.rounds)
    elif a.mode == "profile":
        profile(new, net, a.rounds)
        overhead(new, net)
    else:
        old = loadModule(a.old or gitOld(), "fastEnv_old")
        ab(new, old, net, a.rounds)
