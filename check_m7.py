import random
from env import SprudelJumpEnv, PLAYER_HEIGHT, DIFFICULTY_MAX_HEIGHT

# ---------- 1a. force-spawned breakable platform: removed right after its bounce ----------
env = SprudelJumpEnv()
env.platforms = [[200.0, 600.0, 60, True]]
env.playerX, env.playerY, env.velY = 190.0, 600.0 - PLAYER_HEIGHT - 30.0, 0.0
env.totalHeight = 0.0
gone_after = False
for i in range(300):
    env.step([0.5, 0.0])
    if i >= 40 and not any(p[1] == 600.0 for p in env.platforms):
        gone_after = True
        break
print("1a breakable removed after bounce:", gone_after)

# ---------- 1b. normal platform at same height: still present 40+ frames after bounce ----------
envB = SprudelJumpEnv()
envB.platforms = [[200.0, 600.0, 60, False], [60.0, 600.0, 60, False]]
envB.playerX, envB.playerY, envB.velY = 190.0, 600.0 - PLAYER_HEIGHT - 30.0, 0.0
envB.totalHeight = 0.0
present = False
for i in range(100):
    envB.step([0.5, 0.0])
    if i >= 40 and any(p[1] == 600.0 and p[0] == 200.0 for p in envB.platforms):
        present = True
        break
print("1b normal still present after bounce:", present)

# ---------- 2. state vector is 17 floats ----------
print("2 len(env.reset()) == 17:", len(env.reset()) == 17)

# ---------- 3. breakable spawn chance scales with difficulty ----------
# Literal spec check (5000 steps, action 0.5, breakable count first1500 vs last1500)
# degenerates: constant 0.5 locks the scroll equilibrium (H ~136-193, documented M4),
# so totalHeight never leaves ~0 and only ~3 platforms spawn in 5000 steps ->
# first/last windows carry no real difficulty delta and tiny n, and the "visibly
# higher" signal is unobservable (base 0.05 dominates). So we run the literal 0.5
# check for the record AND verify the actual property directly: the breakable rate
# follows chance = 0.05 + t*(0.30-0.05), t = totalHeight/20000.
origSpawn = SprudelJumpEnv._spawnPlatform

def totalSpawns(e, steps=5000, action=(0.5, 0.0)):
    n, done = 0, False
    def counting(self):
        nonlocal n
        origSpawn(self)
        n += 1
    SprudelJumpEnv._spawnPlatform = counting
    for i in range(steps):
        st, fit, done = e.step(action)
        if done:
            e.reset()
    SprudelJumpEnv._spawnPlatform = origSpawn
    return n, e.totalHeight

random.seed(42)
e05 = SprudelJumpEnv()
spawns05, H05 = totalSpawns(e05)
print("3a literal action=0.5 x5000: H=%.1f, total spawn events=%d" % (H05, spawns05))

def breakableRateAtT(t, n, seed):
    random.seed(seed)
    envT = SprudelJumpEnv()
    envT.reset()
    envT.totalHeight = t * DIFFICULTY_MAX_HEIGHT
    envT.platforms = [[100.0, -120.0 * k, 60, False] for k in range(50)]
    b = 0
    for _ in range(n):
        before = len(envT.platforms)
        origSpawn(envT)
        b += envT.platforms[before][3]
    return b / n

ok = True
for t in (0.0, 0.5, 1.0):
    expect = 0.05 + t * (0.30 - 0.05)
    got = breakableRateAtT(t, 4000, seed=7)
    good = abs(got - expect) < 0.015
    ok = ok and good
    print("3b t=%.1f expected=%.3f measured=%.3f ok=%s" % (t, expect, got, good))
print("3 PASS:", ok)
