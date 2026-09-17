import random
from env import SprudelJumpEnv, BOUNCE_VELOCITY
e = SprudelJumpEnv()
# 3 platforms at different x/y, each with its own width
e.platforms = [[0, 550, 400], [150, 350, 100], [350, 150, 50]]
random.seed(1)
bounces = 0
for i in range(300):
    prev = e.velY
    e.step(random.random())
    if prev > 0 and abs(e.velY - BOUNCE_VELOCITY) < 1e-9:
        bounces += 1
print("bounces:", bounces)
assert 0 < bounces < 50, "expected several, not zero nor every frame"
print("M3 OK")
