import random
from env import (SprudelJumpEnv, PLAYER_HEIGHT, MONSTER_WIDTH, MONSTER_HEIGHT,
                 MONSTER_KILL_BONUS, DIFFICULTY_MAX_HEIGHT, SCROLL_THRESHOLD_Y)

# ---------- 1. stomp: monster above platform, player falls from above ----------
env = SprudelJumpEnv()
env.platforms = [[170.0, 600.0, 60, False]]
env.monsters = [[182.0, 564.0]]  # 36x36, top at 564, right on the platform
env.playerX, env.playerY, env.velY = 190.0, 500.0, 0.0  # falling toward 600 (feet at 540)
env.totalHeight = 0.0
stomped, bounced, died = False, False, False
for i in range(100):
    st, fit, done = env.step([0.5, 0.0])
    if done or env.playerY > 700:
        died = True
        break
    if not env.monsters:
        stomped = True
        bounced = env.velY < 0
        break
print("1 stomp removed monster:", stomped, "| height bonus:", env.totalHeight >= MONSTER_KILL_BONUS, "| bounced:", bounced, "| not died:", not died)

# ---------- 2. lethal contact: rising player hits monster side ----------
env2 = SprudelJumpEnv()
env2.platforms = [[170.0, 750.0, 60, False]]
env2.monsters = [[205.0, 330.0]]  # side of player, player rising (velY < 0 -> stomp branch dead)
env2.playerX, env2.playerY, env2.velY = 200.0, 335.0, -5.0
env2.totalHeight = 0.0
dead, monsterLeft = False, False
for i in range(20):
    st, fit, done = env2.step([0.5, 0.0])
    if done:
        dead = True
        monsterLeft = len(env2.monsters) == 1
        break
print("2 lethal contact done=True:", dead, "| monster NOT removed:", monsterLeft)

# ---------- 3. state length ----------
print("3 len(env.reset()) == 23:", len(env.reset()) == 23)

# ---------- 4. monster spawn ramp, literal action=0.5 first1500 vs last1500 ----------
# Same caveat as M7 check 3: constant 0.5 locks the scroll equilibrium (H stays ~136-200,
# documented M4), so totalHeight never leaves ~0 -> difficultyT stays ~0 -> per the spec
# formula (base 0.0, max 0.35 * t) the spawn chance is ~0 the whole run. No visible ramp
# is observable under action 0.5, so the literal 5000-step count is recorded for the
# record and the ACTUAL property (chance scales with t) is verified directly.
origSpawn = SprudelJumpEnv._spawnPlatform
random.seed(123)
e = SprudelJumpEnv()
for i in range(5000):
    st, fit, done = e.step([0.5, 0.0])
    if done:
        e.reset()
counts_note = "literal action=0.5 x5000: H=%.1f monsters alive=%d" % (e.totalHeight, len(e.monsters))

def rateAtT(t, n, seed):
    random.seed(seed)
    envT = SprudelJumpEnv()
    envT.reset()
    envT.totalHeight = t * DIFFICULTY_MAX_HEIGHT
    envT.monsters = []
    rolled = 0
    for _ in range(n):
        before = len(envT.monsters)
        origSpawn(envT)
        if len(envT.monsters) > before:
            rolled += 1
    return rolled / n

print("4a", counts_note)
ok = True
for t in (0.0, 0.5, 1.0):
    expect = 0.0 + t * (0.35 - 0.0)
    got = rateAtT(t, 4000, seed=7)
    good = abs(got - expect) < 0.02
    ok = ok and good
    print("4b t=%.1f expected=%.3f measured=%.3f ok=%s" % (t, expect, got, good))
print("4 PASS:", ok)
