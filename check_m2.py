from env import SprudelJumpEnv
e = SprudelJumpEnv()
for i in range(1, 101):
    e.step([1.0, 0.0])
    if i % 10 == 0:
        print(i, round(e.playerX, 2), round(e.playerY, 2))
