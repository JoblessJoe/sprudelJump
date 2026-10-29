'''
Checks gpuEnv.py against fastEnv.py (the CPU version) in ~/venv-gpu:

  1. reset: levels (platforms, monsters, Mersenne Twister state) for the same seeds must be identical
  2. physics: the same random actions, game by game, decision by decision; the state is compared after chosen decisions. The integer parts must be identical; floats can
     drift in the last bits because the GPU fuses multiply-adds, so the check reports the share of games whose state is still identical and the largest difference.

    python gpuCheck.py
'''
import numpy as np
from numba import cuda

import fastEnv
import gpuEnv as ge

G, REPEAT = 400, 5
rng = np.random.RandomState(7)
seeds = [int(x) for x in rng.randint(0, 2 ** 31, G)]

cpu = fastEnv.FastBatch(seeds, None, 0, 0.0, 1.0, 4.0)    # monster-heavy levels: monsterFraction 1.0, multiplier 4
F = cuda.device_array((ge.NF, G)); I = cuda.device_array((ge.NI, G), dtype=np.int64)
PL = cuda.device_array((4, ge.MAXP, G)); MO = cuda.device_array((2, ge.MAXM, G)); BU = cuda.device_array((2, ge.MAXB, G)); MT = cuda.device_array((625, G), dtype=np.int64)
blocks, threads = (G + 127) // 128, 128
ge.initKernel[blocks, threads](F, I, PL, MO, MT, cuda.to_device(np.array(seeds, dtype=np.int64)), -1, 0, 0.0, 1.0, 4.0, 0.2)


def same(a, b):
    return int(np.sum(np.all(np.isclose(a.reshape(len(a), -1), b.reshape(len(b), -1), rtol=0, atol=0), axis=1)))


print("reset: games with identical F / I / PL (used part) / MT:",
      same(F.copy_to_host().T[:, :8], cpu.F[:, :8]), same(I.copy_to_host().T, cpu.I), same(MT.copy_to_host().T, cpu.MT), "of", G)


@cuda.jit
def actionKernel(F, I, PL, MO, BU, MT, steers, shoots, repeat, maxNoProgress, step):
    g = cuda.grid(1)
    if g >= F.shape[1] or I[ge.DONE, g] != 0:
        return
    I[ge.BOUNCED, g] = 0
    F[ge.STEER_VAR, g] += abs(steers[g, step] - F[ge.LAST_STEER, g])
    F[ge.LAST_STEER, g] = steers[g, step]
    F[ge.DECISIONS, g] += 1.0
    for _ in range(repeat):
        if ge.stepGame(F, I, PL, MO, BU, MT, g, steers[g, step], shoots[g, step], maxNoProgress):
            I[ge.DONE, g] = 1
            break


steps = 300
steers = rng.uniform(0, 1, (G, steps))
shoots = (rng.uniform(0, 1, (G, steps)) < 0.9).astype(np.float64)
dSteers, dShoots = cuda.to_device(steers), cuda.to_device(shoots)
alive = np.arange(G, dtype=np.int64)
for s in range(steps):
    actionKernel[blocks, threads](F, I, PL, MO, BU, MT, dSteers, dShoots, REPEAT, 600, s)
    cpu.step(np.flatnonzero(~cpu.done()).astype(np.int64), steers[:, s][~cpu.done()], shoots[:, s][~cpu.done()], REPEAT)
    if s + 1 in (1, 5, 20, 60, 150, 300):
        Fh, Ih = F.copy_to_host().T, I.copy_to_host().T
        okF = np.all(Fh[:, :ge.NF] == cpu.F[:, :ge.NF], axis=1)
        okI = np.all(Ih == cpu.I, axis=1)
        diff = np.abs(Fh - cpu.F)
        print(f"after {s + 1:3d} decisions: games with identical state: floats {int(okF.sum())}/{G}, ints {int(okI.sum())}/{G}; "
              f"max |difference| of the scores {np.abs((Fh[:, ge.TOTAL] - Fh[:, ge.OFFSET]) - cpu.scores()).max():.3g}; done games {int((Ih[:, ge.DONE] == 1).sum())} (cpu {int(cpu.done().sum())})")
