'''Strongest identity check for fastEnv changes: the old fastEnv (git HEAD or the commit given as argument) and the current file stepped in lockstep with identical random actions on monster-heavy levels (always shooting); the WHOLE state (F, I, PL, MO, BU, MT) is compared after every decision. Run: python testLockstep.py (the reference commit must have the same game rules; compares states, observations)'''
import os, sys, subprocess, importlib.util
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
REF = sys.argv[1] if len(sys.argv) > 1 else "HEAD"    # the commit whose fastEnv.py is the reference (it must have the same game rules)
src = subprocess.run(["git", "show", REF + ":fastEnv.py"], cwd=HERE, capture_output=True, text=True, check=True).stdout
os.makedirs(HERE + "/bench/_old", exist_ok=True)
open(HERE + "/bench/_old/fastEnv_old.py", "w").write(src)
old = load(HERE + "/bench/_old/fastEnv_old.py", "fastEnv_old"); new = load(HERE + "/fastEnv.py", "fastEnv")
LAYOUT = (3, 2, 1, 2, 0, 1)
total = 0; bad = 0
for case, (mf, mm, mx, mn, z, shootP, steerMode, repeat) in enumerate((
        (1.0, 4.0, None, 0, 0.0, 1.0, "random", 5), (1.0, 6.0, None, 0, 0.0, 0.7, "bang", 5), (0.5, 3.0, 30000, 20000, 0.5, 1.0, "random", 1),
        (1.0, 8.0, 30000, 5000, 0.0, 0.4, "drift", 3), (0.0, 1.0, None, 0, 0.0, 1.0, "random", 5), (1.0, 5.0, None, 0, 0.0, 1.0, "hover", 5))):
    rng = np.random.RandomState(100 + case)
    G = 120
    seeds = [int(x) for x in rng.randint(0, 2 ** 31, G)]
    A = old.FastBatch(seeds, mx, mn, z, mf, mm, stable=LAYOUT); B = new.FastBatch(seeds, mx, mn, z, mf, mm)
    aliveA = np.arange(G, dtype=np.int64); aliveB = aliveA.copy()
    steps = 0
    while len(aliveA) and steps < 4000:
        n = len(aliveA)
        if steerMode == "random": st = rng.uniform(0, 1, n)
        elif steerMode == "bang": st = rng.randint(0, 2, n).astype(float)
        elif steerMode == "drift": st = np.full(n, 0.5) + 0.2 * np.sin(steps / 17.0)
        else: st = np.clip(0.5 + 0.1 * rng.randn(n), 0, 1)
        sh = (rng.uniform(0, 1, n) < shootP).astype(float) * 0.9
        A.step(aliveA, st, sh, repeat); B.step(aliveB, st, sh, repeat)
        # state columns by name (the column layout changed with the removal of the target mode: no ghost columns, no platform ids)
        pairs = ((A.F[:, :8], B.F[:, :8]), (A.F[:, 12:17], B.F[:, 8:13]), (A.I[:, :7], B.I[:, :7]), (A.I[:, 9], B.I[:, 7]), (A.PL[:, :, :4], B.PL),
                 (A.MO, B.MO), (A.BU, B.BU), (A.MT, B.MT), (A.observe(np.arange(G, dtype=np.int64), 33), B.observe(np.arange(G, dtype=np.int64))))
        for a, b_ in pairs:
            if not np.array_equal(a, b_):
                bad += 1; print(f"  !! state differs: case {case} step {steps}"); break
        else:
            aliveA = aliveA[A.I[aliveA, 5] == 0]; aliveB = aliveB[B.I[aliveB, 5] == 0]
            steps += 1; total += len(aliveA)
            continue
        break
    print(f"case {case}: monsters {mf}x{mm}, shoot {shootP}, {steerMode}, repeat {repeat}: {steps} decisions, states identical" if not bad else "FAILED")
print("game-decisions compared:", total, " failures:", bad)
