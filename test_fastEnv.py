'''
Checks fastEnv against env.py (the reference): same seeds -> same levels, same observations, same scores, frame by frame.
    python test_fastEnv.py            (needs numba)
'''
import random

import numpy as np

from env import SprudelJumpEnv
from fastEnv import FastBatch

CONFIGS = [
    dict(maxStart=None, minStart=0, zero=0.0, mf=0.0, mm=1.0),
    dict(maxStart=30000, minStart=20000, zero=0.5, mf=0.0, mm=1.0),
    dict(maxStart=30000, minStart=0, zero=0.0, mf=0.0, mm=1.0),
    dict(maxStart=30000, minStart=20000, zero=0.5, mf=0.3, mm=2.0),
    dict(maxStart=None, minStart=0, zero=0.0, mf=1.0, mm=4.0),    # monster-heavy levels (bullets vs monsters at the top edge)
    dict(maxStart=None, minStart=0, zero=0.0, mf=0.0, mm=1.0, dz=0.0),    # steer dead zone 0 (the start of a run from scratch)
    dict(maxStart=30000, minStart=20000, zero=0.5, mf=0.3, mm=2.0, dz=0.1),
]
POLICIES = {
    "random": lambda rng, f: [rng.random(), 1.0 if rng.random() > 0.5 else 0.0],
    "drift": lambda rng, f: [0.5 + 0.5 * rng.random(), 1.0],
    "bang-bang": lambda rng, f: [1.0 if (f // 7) % 2 else 0.0, 0.0],
}


def compare(cfg, seeds, frames, policy):
    '''Plays the games in both envs with identical actions; asserts equal observations, scores and game-over flags every frame.'''
    envs = [SprudelJumpEnv(steerDeadZone=cfg.get("dz", 0.2)) for _ in seeds]
    states = {g: e.reset(cfg["maxStart"], s, cfg["minStart"], cfg["zero"], cfg["mf"], cfg["mm"]) for g, (e, s) in enumerate(zip(envs, seeds))}
    b = FastBatch(seeds, cfg["maxStart"], cfg["minStart"], cfg["zero"], cfg["mf"], cfg["mm"], deadZone=cfg.get("dz", 0.2))
    alive = list(range(len(seeds)))
    rng = random.Random(1)
    checked = 0
    for f in range(frames):
        if not alive:
            break
        idx = np.array(alive, dtype=np.int64)
        obs = b.observe(idx)
        for row, g in zip(obs, alive):
            assert np.array_equal(row, np.array(states[g])), f"observation differs: game {g} frame {f}\n{row}\n{np.array(states[g])}"
        acts = [policy(rng, f) for _ in alive]
        results = {g: envs[g].step(a) for g, a in zip(alive, acts)}
        b.step(idx, [a[0] for a in acts], [a[1] for a in acts], 1)
        for g in alive:
            st, sc, d = results[g]
            assert b.scores()[g] == sc, f"score differs: game {g} frame {f}: {b.scores()[g]} vs {sc}"
            assert bool(b.done()[g]) == d, f"game-over flag differs: game {g} frame {f}"
            assert b.detours()[g] == envs[g].detour, f"detour differs: game {g} frame {f}: {b.detours()[g]} vs {envs[g].detour}"
            assert b.deviations()[g] == envs[g].deviation, f"deviation differs: game {g} frame {f}: {b.deviations()[g]} vs {envs[g].deviation}"
            assert b.causes()[g] == envs[g].deathCause, f"death cause differs: game {g} frame {f}: {b.causes()[g]} vs {envs[g].deathCause}"
            states[g] = st
        checked += len(alive)
        alive = [g for g in alive if not results[g][2]]
    return checked


if __name__ == "__main__":
    for ci, cfg in enumerate(CONFIGS):
        for name, pol in POLICIES.items():
            seeds = [random.Random(ci).randrange(2**32) for _ in range(40)] + [0, 1, 2, 2**32 - 1]
            n = compare(cfg, seeds, 4000, pol)
            print(f"config {ci} policy {name}: {n:,} game-frames identical")
    print("ALL IDENTICAL")
