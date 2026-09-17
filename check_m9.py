import random
from env import (SprudelJumpEnv, MONSTER_WIDTH, MONSTER_HEIGHT, MONSTER_KILL_BONUS,
                 BULLET_SPEED, BULLET_COOLDOWN_FRAMES, BULLET_WIDTH, BULLET_HEIGHT, PLAYER_WIDTH)

# ---------- 1. bullet kill: monster directly above the bullet stream ----------
env = SprudelJumpEnv()
env.platforms = [[200.0, 600.0, 60, False]]
env.playerX, env.playerY, env.velY = 200.0, 400.0, 0.0
env.monsters = [[202.0, 320.0]]
# bullet spawns at x = playerX + PLAYER_WIDTH/2 - BULLET_WIDTH/2 = 217; center monster on that
env.monsters = [[217 - MONSTER_WIDTH / 2, 320.0]]
env.totalHeight = 0.0
sawBullet, killedAt, movedUp = False, None, False
for i in range(60):
    st, fit, done = env.step([0.5, 1.0])
    if env.bullets:
        sawBullet = True
        movedUp = movedUp or any(b[1] < 386.0 for b in env.bullets)
    if not env.monsters:
        killedAt = fit
        break
print("1 bullet appeared:", sawBullet, "| bullet moved upward:", movedUp,
      "| monster removed:", killedAt is not None,
      "| height jumped to >= KILL_BONUS:", killedAt is not None and killedAt >= MONSTER_KILL_BONUS)

# ---------- 2. cooldown limits fire rate ----------
class Probe(SprudelJumpEnv):
    def __init__(self):
        self.shots = 0
        super().__init__()
    def step(self, action):
        out = super().step(action)
        # after a shot, cooldown was set to BULLET_COOLDOWN_FRAMES then decremented -> == FRAMES-1
        if self.cooldown == BULLET_COOLDOWN_FRAMES - 1:
            self.shots += 1
        return out

env2 = Probe()
env2.platforms = [[200.0, 600.0, 60, False]]
nFrames = BULLET_COOLDOWN_FRAMES * 3
for i in range(nFrames):
    st, fit, done = env2.step([0.5, 1.0])
    if done:
        env2.reset()
expected = 3  # one per cooldown window over 3 windows
ok = abs(env2.shots - expected) <= 1
print("2 shots fired in %d frames: %d (expected ~%d, definitely not %d)" %
      (nFrames, env2.shots, expected, nFrames), "| cooldown works:", ok)
