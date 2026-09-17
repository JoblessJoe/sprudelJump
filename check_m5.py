import random
random.seed(42)
from env import SprudelJumpEnv
env = SprudelJumpEnv()
s = env.reset()
assert len(s) == 12
steps = 0
done = False
while not done and steps < 5000:
    s, fitness, done = env.step(random.random())
    steps += 1
print(steps, fitness)
