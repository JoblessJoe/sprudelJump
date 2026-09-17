import statistics
from env import SprudelJumpEnv, PLATFORM_WIDTH, PLATFORM_WIDTH_MIN, PLATFORM_GAP_MIN, PLATFORM_GAP_MAX, PLATFORM_GAP_MIN_HARD, PLATFORM_GAP_MAX_HARD, DIFFICULTY_MAX_HEIGHT

# --- Part A: M4 self-check as written: 3000 steps, action=0.5 ---
e = SprudelJumpEnv()
first500_spawns, last500_spawns = [], []
orig = e._spawnPlatform
def record():
    orig()
    p = e.platforms[-1]
    topY = min(q[1] for q in e.platforms[:-1])
    (first500_spawns if e.step_n <= 500 else last500_spawns).append((topY - p[1], p[2]))
e._spawnPlatform = record
e.step_n = 0
orig_step = e.step
def step_wrap(action):
    e.step_n += 1
    return orig_step(action)
e.step = step_wrap
prev = 0
ok_monotonic = True
for i in range(1, 3001):
    s, f, d = e.step([0.5, 0.0])
    if i % 300 == 0:
        print(i, "H=" + str(round(f, 1)))
    if f < prev - 1e-9:
        ok_monotonic = False
        print("!! DECREASE at", i)
    prev = f
    if d:
        print("  done at", i)
        e.reset()
def avg(rs, idx):
    v = [r[idx] for r in rs]
    return round(statistics.mean(v), 3) if v else None
print("monotonic:", ok_monotonic)
print("first500 spawns n=%d avg gap=%s avg width=%s" % (len(first500_spawns), avg(first500_spawns, 0), avg(first500_spawns, 1)))
print("last500  spawns n=%d avg gap=%s avg width=%s" % (len(last500_spawns), avg(last500_spawns, 0), avg(last500_spawns, 1)))

# --- Part B: difficulty formula at t = 0, 0.25, 0.5, 1.0 (1000 spawns each) ---
print()
for t_target in (0.0, 0.25, 0.5, 1.0):
    e2 = SprudelJumpEnv()
    e2.totalHeight = t_target * DIFFICULTY_MAX_HEIGHT
    gaps, widths = [], []
    for _ in range(1000):
        before = len(e2.platforms)
        e2._spawnPlatform()
        p = e2.platforms[-1]
        topY = min(q[1] for q in e2.platforms[:-1])
        gaps.append(topY - p[1]); widths.append(p[2])
    exp_gmin = PLATFORM_GAP_MIN + t_target * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
    exp_gmax = PLATFORM_GAP_MAX + t_target * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
    exp_w = PLATFORM_WIDTH - t_target * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
    print("t=%.2f gap=[%.1f,%.1f] expected=[%.1f,%.1f] | width mean=%.2f expected=%.2f"
          % (t_target, min(gaps), max(gaps), exp_gmin, exp_gmax, statistics.mean(widths), exp_w))
