'''
Numba-compiled copy of SprudelJumpEnv for evaluating MANY games at once (one network, many seeds, lockstep).
The Python env (env.py) stays the reference: every rule here is a line-by-line port of it, and the level generator uses
CPython's own Mersenne Twister (seeding, random(), uniform(), randint()), so the same seed gives the same levels and, with
the same actions, the same scores bit for bit (see test_fastEnv.py).

Supported: classic and stableSlots observations, the optional difficulty input, startHeight/zeroFraction/monsterFraction practice modes,
target-mode bookkeeping (a committed target platform per game, kept by id).
Not supported: seed=None (random level) - seeds must be ints in [0, 2**32).

Typical use (see tensorNetwork/proSprudler.evaluateNetworkFast):
    b = FastBatch(seeds, maxStart, minStart, zeroFraction, monsterFraction, monsterMult, stable)
    obs = b.observe(alive, nIn)              # float64 [len(alive), nIn]
    ...                                      # network -> actions
    b.step(alive, steer, shoot, repeat)      # advances every game in 'alive' by 'repeat' frames
    b.scores()                               # points beyond the start, per game
Faster loop with the same results (one compiled call per network decision instead of observe + decide + step + filter):
    alive, obs = b.startLoop(nIn); n = len(alive)
    while n:
        y = network(obs[:n])                 # float32 [n, nOut]
        n = b.stepObserve(alive, n, y, applied, smooth, repeat)
'''
import numpy as np
from numba import njit
try:
    import scipy.linalg.cython_blas    # numba's np.dot calls the BLAS through scipy; without scipy forwardNet cannot be compiled
    HAVE_BLAS = True
except ImportError:
    HAVE_BLAS = False


from env import (SCREEN_WIDTH, HALF_WIDTH, SCREEN_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT, PLATFORM_WIDTH, GRAVITY, BOUNCE_VELOCITY,
                 MAX_FALL_SPEED, HORIZONTAL_SPEED, SCROLL_THRESHOLD_Y, PLATFORM_GAP_MIN, PLATFORM_GAP_MAX, DIFFICULTY_MAX_HEIGHT,
                 PLATFORM_GAP_MIN_HARD, PLATFORM_GAP_MAX_HARD, PLATFORM_WIDTH_MIN, BREAKABLE_CHANCE_BASE, BREAKABLE_CHANCE_MAX,
                 MONSTER_WIDTH, MONSTER_HEIGHT, MONSTER_SPAWN_CHANCE_BASE, MONSTER_SPAWN_CHANCE_MAX, MONSTER_SPAWN_CHANCE_CAP,
                 MONSTER_KILL_BONUS, BULLET_WIDTH, BULLET_HEIGHT, BULLET_SPEED, BULLET_COOLDOWN_FRAMES, PROGRESS_MARGIN)

LAND_TMAX, LAND_TOLERANCE = 100, 40.0    # landingInfo, see env.py
MAXP, MAXM, MAXB = 48, 48, 24          # array capacities per game (a screen holds ~12 platforms, ~7 bullets)
# float state columns F[g, :]
PX, PY, VY, TOTAL, OFFSET, BEST, MMULT = 0, 1, 2, 3, 4, 5, 6
# 'ghost' of a removed target platform (the Python env keeps feeding the frozen last position of the platform object for one frame)
GHOST_FLAG, GHOST_X, GHOST_Y, GHOST_W = 8, 9, 10, 11
TOPY = 7    # y of the topmost platform after the last frame (column 7 was unused)
# int state columns I[g, :]
COOL, FWP, NP, NM, NB, DONE, BOUNCED, NEXTID, TARGET, CAUSE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9   # CAUSE: 0 running, 1 monster while rising, 2 monster otherwise, 3 fell, 4 no progress
# platform columns: x, y, width, breakable, id
MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------- CPython's Mersenne Twister (state: mt[0:624], index in mt[624])
@njit(cache=True)
def mtSeed(mt, seed):
    '''random.Random(seed) for 0 <= seed < 2**32: init_by_array([seed]).'''
    mt[0] = 19650218
    for i in range(1, 624):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & MASK32
    i = 1
    j = 0
    for _ in range(624):
        mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525)) + seed + j) & MASK32
        i += 1
        j += 1
        if i >= 624:
            mt[0] = mt[623]
            i = 1
        if j >= 1:
            j = 0
    for _ in range(623):
        mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i) & MASK32
        i += 1
        if i >= 624:
            mt[0] = mt[623]
            i = 1
    mt[0] = 0x80000000
    mt[624] = 624


@njit(cache=True)
def mtNext(mt):
    '''genrand_uint32'''
    if mt[624] >= 624:
        # the same in-place refill as before, in three parts without the modulo (kk + 1 and kk + 397 wrap only at the end):
        # entries 0..226 read mt[kk + 397] (old values), 227..622 read mt[kk - 227] (already refilled), the last one wraps to mt[0] and mt[396]
        for kk in range(227):
            y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF)
            v = mt[kk + 397] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908B0DF
            mt[kk] = v
        for kk in range(227, 623):
            y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF)
            v = mt[kk - 227] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908B0DF
            mt[kk] = v
        y = (mt[623] & 0x80000000) | (mt[0] & 0x7FFFFFFF)
        v = mt[396] ^ (y >> 1)
        if y & 1:
            v ^= 0x9908B0DF
        mt[623] = v
        mt[624] = 0
    y = mt[mt[624]]
    mt[624] += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & MASK32


@njit(cache=True)
def rndRandom(mt):
    a = mtNext(mt) >> 5
    b = mtNext(mt) >> 6
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def rndUniform(mt, lo, hi):
    return lo + (hi - lo) * rndRandom(mt)


@njit(cache=True)
def rndRandint(mt, lo, hi):
    n = hi - lo + 1
    k = 0
    m = n
    while m > 0:
        k += 1
        m >>= 1
    r = mtNext(mt) >> (32 - k)
    while r >= n:
        r = mtNext(mt) >> (32 - k)
    return lo + r


# ---------------------------------------------------------------- level generation
@njit(cache=True)
def addPlatform(PL, I, g, x, y, w, breakable):
    n = I[g, NP]
    PL[g, n, 0] = x
    PL[g, n, 1] = y
    PL[g, n, 2] = w
    PL[g, n, 3] = breakable
    PL[g, n, 4] = I[g, NEXTID]
    I[g, NEXTID] += 1
    I[g, NP] = n + 1


@njit(cache=True)
def spawnPlatform(F, I, PL, MO, MT, g):
    topY = PL[g, 0, 1]
    for i in range(1, I[g, NP]):
        if PL[g, i, 1] < topY:
            topY = PL[g, i, 1]
    spawnPlatformAbove(F, I, PL, MO, MT, g, topY)


@njit(cache=True)
def spawnPlatformAbove(F, I, PL, MO, MT, g, topY):
    '''Adds the next platform above the topmost one (y = topY, the minimum y of the existing platforms) and returns its y, the new topmost
    (the gap is at least PLATFORM_GAP_MIN > 0, so the new platform is always above all the others).'''
    t = min(1.0, F[g, TOTAL] / DIFFICULTY_MAX_HEIGHT)
    gapMin = PLATFORM_GAP_MIN + t * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
    gapMax = PLATFORM_GAP_MAX + t * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
    width = PLATFORM_WIDTH - t * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
    breakable = 1.0 if rndRandom(MT[g]) < (BREAKABLE_CHANCE_BASE + t * (BREAKABLE_CHANCE_MAX - BREAKABLE_CHANCE_BASE)) else 0.0
    x = rndUniform(MT[g], 0.0, SCREEN_WIDTH - width)
    y = topY - rndUniform(MT[g], gapMin, gapMax)
    addPlatform(PL, I, g, x, y, width, breakable)
    if rndRandom(MT[g]) < min(MONSTER_SPAWN_CHANCE_CAP, (MONSTER_SPAWN_CHANCE_BASE + t * (MONSTER_SPAWN_CHANCE_MAX - MONSTER_SPAWN_CHANCE_BASE)) * F[g, MMULT]):
        n = I[g, NM]
        MO[g, n, 0] = x + width / 2 - MONSTER_WIDTH / 2
        MO[g, n, 1] = y - MONSTER_HEIGHT
        I[g, NM] = n + 1
    return y


@njit(cache=True)
def resetGame(F, I, PL, MO, MT, g, seed, maxStart, minStart, zeroFraction, monsterFraction, monsterMult):
    '''maxStart < 0 means None. Same sequence of random numbers as SprudelJumpEnv.reset.'''
    mtSeed(MT[g], seed)
    F[g, PX] = SCREEN_WIDTH / 2
    F[g, PY] = SCREEN_HEIGHT / 2
    F[g, VY] = 0.0
    offset = 0.0
    if maxStart >= 0:
        zero = False
        if zeroFraction > 0:
            zero = rndRandom(MT[g]) < zeroFraction
        if not zero:
            offset = float(rndRandint(MT[g], minStart, maxStart))
    F[g, OFFSET] = offset
    F[g, TOTAL] = offset
    F[g, MMULT] = 1.0
    if monsterFraction > 0:
        if rndRandom(MT[g]) < monsterFraction:
            F[g, MMULT] = monsterMult
    I[g, COOL] = 0
    I[g, FWP] = 0
    I[g, NP] = 0
    I[g, NM] = 0
    I[g, NB] = 0
    I[g, DONE] = 0
    I[g, BOUNCED] = 0
    I[g, NEXTID] = 0
    I[g, TARGET] = -1
    F[g, GHOST_FLAG] = 0.0
    t = min(1.0, F[g, TOTAL] / DIFFICULTY_MAX_HEIGHT)
    gapMin = PLATFORM_GAP_MIN + t * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
    gapMax = PLATFORM_GAP_MAX + t * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
    width = PLATFORM_WIDTH - t * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
    y = float(SCREEN_HEIGHT)
    while y > 0:
        x = rndUniform(MT[g], 0.0, SCREEN_WIDTH - width)
        addPlatform(PL, I, g, x, y, width, 0.0)
        y -= rndUniform(MT[g], gapMin, gapMax)
    addPlatform(PL, I, g, F[g, PX] - PLATFORM_WIDTH / 2, F[g, PY] + PLAYER_HEIGHT + 10, PLATFORM_WIDTH, 0.0)
    F[g, BEST] = F[g, TOTAL] + (SCREEN_HEIGHT - F[g, PY])
    topY = PL[g, 0, 1]
    for i in range(1, I[g, NP]):
        if PL[g, i, 1] < topY:
            topY = PL[g, i, 1]
    F[g, TOPY] = topY


@njit(cache=True)
def resetAll(F, I, PL, MO, MT, seeds, maxStart, minStart, zeroFraction, monsterFraction, monsterMult):
    for g in range(len(seeds)):
        resetGame(F, I, PL, MO, MT, g, seeds[g], maxStart, minStart, zeroFraction, monsterFraction, monsterMult)


# ---------------------------------------------------------------- one frame of one game (port of SprudelJumpEnv.step)
@njit(cache=True)
def stepGame(F, I, PL, MO, BU, MT, g, steer, shoot, maxNoProgress):
    '''Advances game g by one frame. Returns True if the game is over.'''
    px = F[g, PX] + (steer - 0.5) * 2 * HORIZONTAL_SPEED
    if px < -PLAYER_WIDTH:
        px = float(SCREEN_WIDTH)
    if px > SCREEN_WIDTH:
        px = float(-PLAYER_WIDTH)
    F[g, PX] = px
    velY = min(F[g, VY] + GRAVITY, MAX_FALL_SPEED)
    feetPrev = F[g, PY] + PLAYER_HEIGHT
    playerY = F[g, PY] + velY
    feet = playerY + PLAYER_HEIGHT
    nP = I[g, NP]
    candidate = False
    if velY > 0:    # most falling frames have no platform inside the band [feetPrev, feet]: look for one before running the full loop below
        for i in range(nP):
            py = PL[g, i, 1]
            if feetPrev <= py and py <= feet:
                candidate = True
                break
    if candidate:
        k = 0
        for i in range(nP):
            py = PL[g, i, 1]
            drop = False
            if feetPrev <= py and py <= feet and px < PL[g, i, 0] + PL[g, i, 2] and px + PLAYER_WIDTH > PL[g, i, 0]:
                velY = BOUNCE_VELOCITY
                I[g, BOUNCED] = 1
                if PL[g, i, 3] > 0.5:
                    drop = True
                    if PL[g, i, 4] == I[g, TARGET]:
                        F[g, GHOST_FLAG] = 1.0
                        F[g, GHOST_X] = PL[g, i, 0]
                        F[g, GHOST_Y] = PL[g, i, 1]
                        F[g, GHOST_W] = PL[g, i, 2]
            if not drop:
                if k != i:
                    for c in range(5):
                        PL[g, k, c] = PL[g, i, c]
                k += 1
        removed = k != nP    # a breakable platform broke: the topmost platform may have changed
        nP = k
        I[g, NP] = k
    else:
        removed = False
    nM = I[g, NM]
    fatal = False
    total = F[g, TOTAL]
    k = 0
    for i in range(nM):
        mx = MO[g, i, 0]
        my = MO[g, i, 1]
        overlapX = px < mx + MONSTER_WIDTH and px + PLAYER_WIDTH > mx
        if velY > 0 and feetPrev <= my and my <= feet and overlapX:
            total += MONSTER_KILL_BONUS
            velY = BOUNCE_VELOCITY
            I[g, BOUNCED] = 1
            continue
        elif overlapX and playerY < my + MONSTER_HEIGHT and playerY + PLAYER_HEIGHT > my:
            fatal = True
        if k != i:
            MO[g, k, 0] = mx
            MO[g, k, 1] = my
        k += 1
    nM = k
    # bullets
    nB = I[g, NB]
    if shoot > 0.5 and I[g, COOL] == 0:
        BU[g, nB, 0] = px + PLAYER_WIDTH / 2 - BULLET_WIDTH / 2
        BU[g, nB, 1] = playerY
        nB += 1
        I[g, COOL] = BULLET_COOLDOWN_FRAMES
    I[g, COOL] = max(0, I[g, COOL] - 1)
    k = 0
    for i in range(nB):
        by = BU[g, i, 1] - BULLET_SPEED
        if by >= -BULLET_HEIGHT:
            BU[g, k, 0] = BU[g, i, 0]
            BU[g, k, 1] = by
            k += 1
    nB = k
    bulletGone = 0
    monsterHit = 0
    for i in range(nB):
        bx = BU[g, i, 0]
        by = BU[g, i, 1]
        for j in range(nM):
            if (bulletGone >> i) & 1 or (monsterHit >> j) & 1:
                continue
            mx = MO[g, j, 0]
            my = MO[g, j, 1]
            if bx < mx + MONSTER_WIDTH and bx + BULLET_WIDTH > mx and by < my + MONSTER_HEIGHT and by + BULLET_HEIGHT > my:
                bulletGone |= 1 << i
                monsterHit |= 1 << j
                total += MONSTER_KILL_BONUS
    if bulletGone != 0:
        k = 0
        for i in range(nB):
            if not (bulletGone >> i) & 1:
                BU[g, k, 0] = BU[g, i, 0]
                BU[g, k, 1] = BU[g, i, 1]
                k += 1
        nB = k
        k = 0
        for j in range(nM):
            if not (monsterHit >> j) & 1:
                MO[g, k, 0] = MO[g, j, 0]
                MO[g, k, 1] = MO[g, j, 1]
                k += 1
        nM = k
    # scrolling. Platforms only move when the screen scrolls, so only then can one fall off the bottom: the scroll, the drop (order
    # preserved) and the search for the topmost platform share ONE pass over the platforms.
    scrolled = playerY < SCROLL_THRESHOLD_Y
    if scrolled:
        scroll = SCROLL_THRESHOLD_Y - playerY
        total += scroll
        playerY = SCROLL_THRESHOLD_Y
        for i in range(nM):
            MO[g, i, 1] += scroll
        for i in range(nB):
            BU[g, i, 1] += scroll
        k = 0
        minY = np.inf
        for i in range(nP):
            y = PL[g, i, 1] + scroll
            if y <= SCREEN_HEIGHT:
                if k != i:
                    PL[g, k, 0] = PL[g, i, 0]
                    PL[g, k, 2] = PL[g, i, 2]
                    PL[g, k, 3] = PL[g, i, 3]
                    PL[g, k, 4] = PL[g, i, 4]
                PL[g, k, 1] = y
                if y < minY:
                    minY = y
                k += 1
            elif PL[g, i, 4] == I[g, TARGET]:
                F[g, GHOST_FLAG] = 1.0
                F[g, GHOST_X] = PL[g, i, 0]
                F[g, GHOST_Y] = y
                F[g, GHOST_W] = PL[g, i, 2]
        nP = k
    elif removed:
        minY = np.inf
        for i in range(nP):
            if PL[g, i, 1] < minY:
                minY = PL[g, i, 1]
    else:
        minY = F[g, TOPY]
    k = 0
    for i in range(nM):
        if MO[g, i, 1] <= SCREEN_HEIGHT:
            if k != i:
                MO[g, k, 0] = MO[g, i, 0]
                MO[g, k, 1] = MO[g, i, 1]
            k += 1
    nM = k
    k = 0
    for i in range(nB):
        if BU[g, i, 1] <= SCREEN_HEIGHT:
            if k != i:
                BU[g, k, 0] = BU[g, i, 0]
                BU[g, k, 1] = BU[g, i, 1]
            k += 1
    nB = k
    I[g, NP] = nP
    I[g, NM] = nM
    I[g, NB] = nB
    F[g, VY] = velY
    F[g, PY] = playerY
    F[g, TOTAL] = total
    # new platforms at the top (minY: the topmost platform, kept in F[g, TOPY] between frames when nothing changed it)
    while I[g, NP] > 0:
        if not minY > 0:
            break
        minY = spawnPlatformAbove(F, I, PL, MO, MT, g, minY)
    F[g, TOPY] = minY
    altitude = F[g, TOTAL] + (SCREEN_HEIGHT - F[g, PY])
    if altitude > F[g, BEST] + PROGRESS_MARGIN:
        F[g, BEST] = altitude
        I[g, FWP] = 0
    else:
        I[g, FWP] += 1
    stuck = maxNoProgress >= 0 and I[g, FWP] >= maxNoProgress
    fell = F[g, PY] > SCREEN_HEIGHT
    I[g, CAUSE] = (1 if velY < 0 else 2) if fatal else (3 if fell else (4 if stuck else 0))
    return fatal or fell or stuck


@njit(cache=True)
def stepMany(F, I, PL, MO, BU, MT, alive, steer, shoot, repeat, maxNoProgress):
    '''Every game in 'alive' plays 'repeat' frames with the same action (stops early when it is over).
    DONE is set; BOUNCED is true if ANY of the frames bounced.'''
    for a in range(len(alive)):
        g = alive[a]
        I[g, BOUNCED] = 0
        for _ in range(repeat):
            if stepGame(F, I, PL, MO, BU, MT, g, steer[a], shoot[a], maxNoProgress):
                I[g, DONE] = 1
                break


# ---------------------------------------------------------------- observation
@njit(cache=True)
def relXY(F, g, x, w):
    '''(relative x of a platform/monster center, wrap-aware, / SCREEN_WIDTH), same convention as env.py'''
    dx = x + w / 2 - (F[g, PX] + PLAYER_WIDTH / 2)
    if dx > HALF_WIDTH:
        dx -= SCREEN_WIDTH
    elif dx < -HALF_WIDTH:
        dx += SCREEN_WIDTH
    return dx / SCREEN_WIDTH


@njit(cache=True)
def clampRel(feet, y):
    ry = (feet - y) / SCREEN_HEIGHT
    return -1.0 if ry < -1.0 else (1.0 if ry > 1.0 else ry)


@njit(cache=True)
def pickNearest(Y, n, key, used, below):
    '''Index of the entry with the smallest key (first one on ties) among not-yet-used entries; below: 1 = only y >= feet, 2 = only y < feet, 0 = all.
    key is computed by the caller; returns -1 if there is none. (A tiny selection sort replaces Python's stable sorted().)'''
    best = -1
    for i in range(n):
        if used[i]:
            continue
        if best < 0 or key[i] < key[best]:
            best = i
    return best


@njit(cache=True)
def slotIds(F, I, PL, g, out, pb, pa):
    '''Platform indices (into PL[g]) of the stableSlots: the pb nearest below the feet (smallest y >= feet first), then the pa nearest above (largest y < feet first); -1 = none.
    One pass over the platforms, each inserted into two small sorted lists; on equal y the lower index stays first (= the order of repeated "pick the nearest unused" scans).'''
    feet = F[g, PY] + PLAYER_HEIGHT
    nP = I[g, NP]
    for s in range(pb + pa):
        out[s] = -1
    cb = 0
    ca = 0
    for i in range(nP):
        y = PL[g, i, 1]
        if y >= feet:
            if cb < pb:
                p = cb
                cb += 1
            elif pb > 0 and y < PL[g, out[pb - 1], 1]:
                p = pb - 1
            else:
                continue
            while p > 0 and PL[g, out[p - 1], 1] > y:
                out[p] = out[p - 1]
                p -= 1
            out[p] = i
        else:
            if ca < pa:
                p = pb + ca
                ca += 1
            elif pa > 0 and y > PL[g, out[pb + pa - 1], 1]:
                p = pb + pa - 1
            else:
                continue
            while p > pb and PL[g, out[p - 1], 1] < y:
                out[p] = out[p - 1]
                p -= 1
            out[p] = i


@njit(cache=True)
def monsterIds(F, I, MO, g, out, mb, ma):
    '''Same for monsters: the mb nearest below the feet, then the ma nearest above; -1 = none.'''
    feet = F[g, PY] + PLAYER_HEIGHT
    nM = I[g, NM]
    for s in range(mb + ma):
        out[s] = -1
    cb = 0
    ca = 0
    for i in range(nM):
        y = MO[g, i, 1]
        if y >= feet:
            if cb < mb:
                p = cb
                cb += 1
            elif mb > 0 and y < MO[g, out[mb - 1], 1]:
                p = mb - 1
            else:
                continue
            while p > 0 and MO[g, out[p - 1], 1] > y:
                out[p] = out[p - 1]
                p -= 1
            out[p] = i
        else:
            if ca < ma:
                p = mb + ca
                ca += 1
            elif ma > 0 and y > MO[g, out[mb + ma - 1], 1]:
                p = mb + ma - 1
            else:
                continue
            while p > mb and MO[g, out[p - 1], 1] < y:
                out[p] = out[p - 1]
                p -= 1
            out[p] = i


@njit(cache=True)
def landFrames(vy, rel):
    '''Frames until the feet land on a platform whose height relative to the feet is rel (= feet - platform y), 0 = none within LAND_TMAX (as env.py).'''
    v = vy
    for t in range(1, LAND_TMAX + 1):
        v = min(v + GRAVITY, MAX_FALL_SPEED)
        prev = rel
        rel += v
        if v > 0:
            if prev <= 0 <= rel:
                return t
            if rel > 0:      # falling and already below the platform: rel only grows from here, it can never land
                return 0
    return 0


LAND_EPS = 1e-6    # far above the rounding gap between the closed form below and landFrames' running sum (~1e-12), far below any real distance


@njit(cache=True)
def landPath(vy):
    '''The fall path shared by all platforms of one frame, in closed form: v_k = vy + k * GRAVITY until it reaches MAX_FALL_SPEED (after K frames).
    Returns (t0, K, A, Z), t0 = first frame with v > 0; t0 = -1 if some v_k is within LAND_EPS of 0 (then the sign of v is a rounding question).
    A = pathSum(t0 - 1), Z = pathSum(LAND_TMAX): the same for every platform of this frame, so they are computed once here.'''
    K = int((MAX_FALL_SPEED - vy) / GRAVITY)
    if K < 0:
        K = 0
    k0 = -vy / GRAVITY
    r = int(round(k0))
    if r >= 1 and abs(vy + r * GRAVITY) < LAND_EPS:
        return -1, K, 0.0, 0.0
    t0 = int(np.floor(k0)) + 1 if vy < 0 else 1
    return t0, K, pathSum(vy, K, t0 - 1), pathSum(vy, K, LAND_TMAX)


@njit(cache=True)
def pathSum(vy, K, t):
    '''v_1 + ... + v_t of landPath's path (equals landFrames' running sum up to rounding).'''
    if t <= K:
        return t * vy + GRAVITY * t * (t + 1) / 2
    return K * vy + GRAVITY * K * (K + 1) / 2 + (t - K) * MAX_FALL_SPEED


@njit(cache=True)
def landFramesPathChecked(vy, rel, t0, K, A, Z):
    '''Slow path of landFramesPath. Same result as landFrames(vy, rel), without the frame loop: from t0 on rel only grows, so it is the first t >= t0 with
    rel + pathSum(t) >= 0 (0 if rel + pathSum(t0 - 1) > 0: already below the platform when it starts falling), found with the quadratic
    formula. The closed form differs from landFrames' running sum by rounding only, so whenever a deciding value is within LAND_EPS of 0
    the exact loop decides instead.'''
    if t0 < 0:
        return landFrames(vy, rel)
    if t0 > LAND_TMAX:
        return 0
    before = rel + A
    if abs(before) < LAND_EPS:
        return landFrames(vy, rel)
    if before > 0:
        return 0
    last = rel + Z
    if abs(last) < LAND_EPS:
        return landFrames(vy, rel)
    if last < 0:
        return 0
    # first guess: larger root of GRAVITY/2 t^2 + (vy + GRAVITY/2) t + rel = 0, or the straight part after K
    b = vy + GRAVITY / 2
    d = b * b - 2 * GRAVITY * rel
    t = int(np.ceil((-b + np.sqrt(d)) / GRAVITY)) if d >= 0 else t0
    if t > K:
        t = K + int(np.ceil(-(rel + pathSum(vy, K, K)) / MAX_FALL_SPEED))
    t = min(max(t, t0), LAND_TMAX)
    while t > t0 and rel + pathSum(vy, K, t - 1) >= 0:    # the guess is off by at most one or two frames
        t -= 1
    while rel + pathSum(vy, K, t) < 0:
        t += 1
    if abs(rel + pathSum(vy, K, t)) < LAND_EPS or (t > t0 and abs(rel + pathSum(vy, K, t - 1)) < LAND_EPS):
        return landFrames(vy, rel)
    return t


@njit(cache=True)
def landFramesPath(vy, rel, t0, K, A, Z):
    '''Same result as landFrames(vy, rel) (and as landFramesPathChecked), faster: the quadratic-formula guess is verified with ONE pathSum
    (s = rel + S_t, and S_(t-1) = S_t - v_t) instead of the up-to-five calls of the checked version. The guess is accepted only when
    s >= 0 and (t == t0 or rel + S_(t-1) < 0) with both values more than LAND_EPS away from 0; every other case (a corrected guess, a value
    within LAND_EPS of 0) goes to landFramesPathChecked, which corrects the guess and falls back to the exact loop near boundaries.
    t == t0 needs no check of S_(t-1): it is 'before' (rel + A), already known to be < -LAND_EPS at that point.'''
    if t0 < 0:
        return landFrames(vy, rel)
    if t0 > LAND_TMAX:
        return 0
    before = rel + A
    if before > -LAND_EPS:
        if before < LAND_EPS:
            return landFrames(vy, rel)
        return 0
    last = rel + Z
    if last < LAND_EPS:
        if last > -LAND_EPS:
            return landFrames(vy, rel)
        return 0
    b = vy + GRAVITY / 2
    d = b * b - 2 * GRAVITY * rel
    t = int(np.ceil((-b + np.sqrt(d)) / GRAVITY)) if d >= 0 else t0
    if t > K:
        t = K + int(np.ceil(-(rel + pathSum(vy, K, K)) / MAX_FALL_SPEED))
    t = min(max(t, t0), LAND_TMAX)
    s = rel + pathSum(vy, K, t)
    if s >= LAND_EPS and (t == t0 or s - (vy + t * GRAVITY if t <= K else MAX_FALL_SPEED) < -LAND_EPS):
        return t
    return landFramesPathChecked(vy, rel, t0, K, A, Z)


@njit(cache=True)
def slotExtras(F, I, PL, MO, g, row, c, i, feet, occupied, landing, t0, K, A, Z):
    '''Optional per-platform features of platform index i (-1 = empty slot): occupied flag, landing prediction [T, r]. Returns the new column.
    t0, K: this frame's landPath (only used with landing).'''
    if i < 0:
        if occupied:
            row[c] = 0.0
            c += 1
        if landing:
            row[c] = 1.0
            row[c + 1] = 1.0
            c += 2
        return c
    if occupied:
        busy = 0.0
        for m in range(I[g, NM]):
            if abs(MO[g, m, 0] + MONSTER_WIDTH / 2 - (PL[g, i, 0] + PL[g, i, 2] / 2)) < 0.5 and abs(MO[g, m, 1] + MONSTER_HEIGHT - PL[g, i, 1]) < 0.5:
                busy = 1.0
                break
        row[c] = busy
        c += 1
    if landing:
        frames = landFramesPath(F[g, VY], feet - PL[g, i, 1], t0, K, A, Z)
        if frames == 0:
            row[c] = 1.0
            row[c + 1] = 1.0
        else:
            dx = PL[g, i, 0] + PL[g, i, 2] / 2 - (F[g, PX] + PLAYER_WIDTH / 2)
            if dx > HALF_WIDTH:
                dx -= SCREEN_WIDTH
            elif dx < -HALF_WIDTH:
                dx += SCREEN_WIDTH
            need = abs(dx) - LAND_TOLERANCE
            if need < 0.0:
                need = 0.0
            r = need / (HORIZONTAL_SPEED * frames)
            row[c] = frames / LAND_TMAX
            row[c + 1] = (2.0 if r > 2.0 else r) / 2.0
        c += 2
    return c


@njit(cache=True)
def observeGame(F, I, PL, MO, g, row, stable, difficulty, pb, pa, mb, ma, occupied, landing, ids, mids):
    '''Writes the 23 (or 24 with difficulty; with stable slots 2 + 3 * (pb + pa) + 2 * (mb + ma) + difficulty) state floats of game g into row; returns the number written.'''
    feet = F[g, PY] + PLAYER_HEIGHT
    vy = F[g, VY] / MAX_FALL_SPEED
    row[0] = F[g, PX] / SCREEN_WIDTH
    row[1] = -1.0 if vy < -1.0 else (1.0 if vy > 1.0 else vy)
    nP = I[g, NP]
    nM = I[g, NM]
    c = 2
    if stable:
        slotIds(F, I, PL, g, ids, pb, pa)
        for s in range(pb + pa):
            i = ids[s]
            if i < 0:
                row[c] = 0.0
                row[c + 1] = -1.0 if s < pb else 1.0
                row[c + 2] = 0.0
            else:
                row[c] = relXY(F, g, PL[g, i, 0], PL[g, i, 2])
                row[c + 1] = clampRel(feet, PL[g, i, 1])
                row[c + 2] = 1.0 if PL[g, i, 3] > 0.5 else 0.0
            c += 3
        monsterIds(F, I, MO, g, mids, mb, ma)
        for s in range(mb + ma):
            i = mids[s]
            if i < 0:
                row[c] = 0.0
                row[c + 1] = -1.0 if s < mb else 1.0
            else:
                row[c] = relXY(F, g, MO[g, i, 0], MONSTER_WIDTH)
                row[c + 1] = clampRel(feet, MO[g, i, 1])
            c += 2
        if occupied or landing:
            t0, K, A, Z = landPath(F[g, VY]) if landing else (0, 0, 0.0, 0.0)
            for s2 in range(pb + pa):
                c = slotExtras(F, I, PL, MO, g, row, c, ids[s2], feet, occupied, landing, t0, K, A, Z)
    else:
        used = np.zeros(nP, dtype=np.bool_)
        key = np.empty(nP, dtype=np.float64)
        for i in range(nP):
            key[i] = abs(feet - PL[g, i, 1])
        for s in range(5):
            i = pickNearest(key, nP, key, used, 0)
            if i < 0:
                row[c] = 0.0
                row[c + 1] = -1.0
                row[c + 2] = 0.0
            else:
                used[i] = True
                row[c] = relXY(F, g, PL[g, i, 0], PL[g, i, 2])
                row[c + 1] = clampRel(feet, PL[g, i, 1])
                row[c + 2] = 1.0 if PL[g, i, 3] > 0.5 else 0.0
            c += 3
        usedM = np.zeros(nM, dtype=np.bool_)
        keyM = np.empty(nM, dtype=np.float64)
        for i in range(nM):
            keyM[i] = abs(feet - MO[g, i, 1])
        for s in range(3):
            i = pickNearest(keyM, nM, keyM, usedM, 0)
            if i < 0:
                row[c] = 0.0
                row[c + 1] = -1.0
            else:
                usedM[i] = True
                row[c] = relXY(F, g, MO[g, i, 0], MONSTER_WIDTH)
                row[c + 1] = clampRel(feet, MO[g, i, 1])
            c += 2
    if difficulty:
        t = F[g, TOTAL] / DIFFICULTY_MAX_HEIGHT
        row[c] = 1.0 if t > 1.0 else t
        c += 1
    return c


@njit(cache=True)
def findPlatform(I, PL, g, pid):
    for i in range(I[g, NP]):
        if PL[g, i, 4] == pid:
            return i
    return -1


@njit(cache=True)
def observeMany(F, I, PL, MO, alive, out, stable, difficulty, targetMode, pb, pa, mb, ma, occupied, landing):
    '''Fills out[a] with the observation of game alive[a]. In target mode (out has 25 columns) the last two are the committed
    target's relative x / signed relative y ([0, -1] without a target).'''
    ids = np.empty(max(pb + pa, 1), dtype=np.int64)
    mids = np.empty(max(mb + ma, 1), dtype=np.int64)
    for a in range(len(alive)):
        g = alive[a]
        c = observeGame(F, I, PL, MO, g, out[a], stable, difficulty, pb, pa, mb, ma, occupied, landing, ids, mids)
        if targetMode:
            i = findPlatform(I, PL, g, I[g, TARGET]) if I[g, TARGET] >= 0 else -1
            if i >= 0:
                out[a, c] = relXY(F, g, PL[g, i, 0], PL[g, i, 2])
                out[a, c + 1] = clampRel(F[g, PY] + PLAYER_HEIGHT, PL[g, i, 1])
            elif F[g, GHOST_FLAG] > 0.5 and I[g, TARGET] >= 0:
                out[a, c] = relXY(F, g, F[g, GHOST_X], F[g, GHOST_W])
                out[a, c + 1] = clampRel(F[g, PY] + PLAYER_HEIGHT, F[g, GHOST_Y])
            else:
                out[a, c] = 0.0
                out[a, c + 1] = -1.0


@njit(cache=True)
def decideTargets(F, I, PL, alive, votes):
    '''Target mode: for every game that needs a new target (it just bounced, it has none, it is gone, or it was passed) the platform slot
    with the highest vote among the platforms in the direction of travel (rising: the 2 above, falling: the 3 below; any existing slot if none
    there) becomes the committed target. votes[a, k] = vote of the network for slot k.'''
    ids = np.empty(5, dtype=np.int64)
    for a in range(len(alive)):
        g = alive[a]
        feet = F[g, PY] + PLAYER_HEIGHT
        t = I[g, TARGET]
        i = findPlatform(I, PL, g, t) if t >= 0 else -1
        need = I[g, BOUNCED] == 1 or t < 0 or i < 0
        if not need:
            ty = PL[g, i, 1]
            need = (F[g, VY] > 0 and ty < feet) or (F[g, VY] < 0 and ty >= feet)
        if not need:
            continue
        F[g, GHOST_FLAG] = 0.0
        slotIds(F, I, PL, g, ids, 3, 2)
        lo = 3 if F[g, VY] < 0 else 0
        hi = 5 if F[g, VY] < 0 else 3
        best = -1
        for k in range(lo, hi):
            if ids[k] >= 0 and (best < 0 or votes[a, k] > votes[a, best]):
                best = k
        if best < 0:
            for k in range(5):
                if ids[k] >= 0 and (best < 0 or votes[a, k] > votes[a, best]):
                    best = k
        I[g, TARGET] = PL[g, ids[best], 4] if best >= 0 else -1


@njit(cache=True)
def stepObserveMany(F, I, PL, MO, BU, MT, alive, n, y, applied, smooth, repeat, maxNoProgress, decide, out,
                    stable, difficulty, targetMode, pb, pa, mb, ma, occupied, landing):
    '''Everything between two forward passes in one call: y[a] = network output of game alive[a] (a < n). decide (target mode): the votes
    y[:, 2:7] pick targets. Steer smoothing (smooth < 1: applied[g] += smooth * (steer - applied[g])), 'repeat' frames, then the finished games are
    dropped from alive[:n] (order kept) and the observation of the remaining ones is written to out[:k] as float32. Returns k.
    Same arithmetic as observe -> astype(float32) -> decide -> smoothing -> step -> filter, so the scores stay bit-identical.'''
    if decide:
        votes = np.empty((n, 5))
        for a in range(n):
            for j in range(5):
                votes[a, j] = np.float64(y[a, 2 + j])
        decideTargets(F, I, PL, alive[:n], votes)
    k = 0
    for a in range(n):
        g = alive[a]
        steer = np.float64(y[a, 0])
        if smooth < 1.0:
            applied[g] += smooth * (steer - applied[g])
            steer = applied[g]
        shoot = np.float64(y[a, 1])
        I[g, BOUNCED] = 0
        for _ in range(repeat):
            if stepGame(F, I, PL, MO, BU, MT, g, steer, shoot, maxNoProgress):
                I[g, DONE] = 1
                break
        if I[g, DONE] == 0:
            alive[k] = g
            k += 1
    observeFloat32(F, I, PL, MO, alive, k, out, stable, difficulty, targetMode, pb, pa, mb, ma, occupied, landing)
    return k


@njit(cache=True)
def observeFloat32(F, I, PL, MO, alive, n, out, stable, difficulty, targetMode, pb, pa, mb, ma, occupied, landing):
    '''observeMany for alive[:n] straight into the float32 array out[:n]. The observation is computed in float64 and rounded on the store, the same
    rounding as np.float32(x) of the float64 value; every column is written for every game, so no zeroing is needed.'''
    observeMany(F, I, PL, MO, alive[:n], out, stable, difficulty, targetMode, pb, pa, mb, ma, occupied, landing)


@njit(cache=True)
def forwardLogits(x, ws, bs):
    '''The network forward pass up to the last layer's pre-activation inside compiled code: x [n, nIn] float32, ws = tuple of weight matrices already
    transposed to [in, out] (float32, C order), bs = tuple of bias vectors; ReLU on the hidden layers. Same BLAS matmuls as numpy's `x @ w`, without ~10 numpy
    calls per decision. The sigmoid of the output stays in numpy on purpose: numpy's float32 exp is not the same function as numba's, a compiled sigmoid
    would differ in the last digit for ~4% of the outputs and the games would diverge.'''
    last = len(ws) - 1
    for L in range(len(ws)):
        x = np.dot(x, ws[L]) + bs[L]
        if L < last:
            x = np.maximum(x, np.float32(0.0))
    return x


class FastBatch:
    '''G games (one per seed) held in arrays, played in lockstep by the compiled functions above.'''

    def __init__(self, seeds, maxStartHeight=None, minStartHeight=0, zeroFraction=0.0, monsterFraction=0.0, monsterMult=1.0,
                 stable=False, difficulty=False, maxFramesWithoutProgress=600):
        G = len(seeds)
        self.G = G
        # stable: False = classic, True = the 3/2/1/2 layout, or a tuple (platformsBelow, platformsAbove, monstersBelow, monstersAbove)
        self.layout = (3, 2, 1, 2) if stable is True else (tuple(stable) if stable else (3, 2, 1, 2))
        self.stable, self.difficulty = bool(stable), difficulty
        self.customLayout = bool(stable) and self.layout != (3, 2, 1, 2)    # 4 or 6 entries (the last two switch the optional extra features on)
        self.occupied = self.stable and len(self.layout) > 4 and bool(self.layout[4])
        self.landing = self.stable and len(self.layout) > 5 and bool(self.layout[5])
        self.obsLength = (2 + 3 * (self.layout[0] + self.layout[1]) + 2 * (self.layout[2] + self.layout[3])
                          + (self.layout[0] + self.layout[1]) * ((1 if self.occupied else 0) + (2 if self.landing else 0))) if self.stable else 23
        self.maxNoProgress = -1 if maxFramesWithoutProgress is None else maxFramesWithoutProgress
        self.F = np.zeros((G, 12))
        self.I = np.zeros((G, 10), dtype=np.int64)
        self.PL = np.zeros((G, MAXP, 5))
        self.MO = np.zeros((G, MAXM, 2))
        self.BU = np.zeros((G, MAXB, 2))
        self.MT = np.zeros((G, 625), dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.int64)
        assert (seeds >= 0).all() and (seeds < 2**32).all(), "seeds must be in [0, 2**32)"
        resetAll(self.F, self.I, self.PL, self.MO, self.MT, seeds, -1 if maxStartHeight is None else int(maxStartHeight), int(minStartHeight),
                 float(zeroFraction), float(monsterFraction), float(monsterMult))

    def observeMode(self, nIn):
        '''(stable, difficulty, targetMode, platformsBelow, platformsAbove, monstersBelow, monstersAbove, occupied, landing) for observeMany.'''
        if self.customLayout:    # other slot counts: nIn is just obsLength (+1 with the difficulty input); no target mode
            assert nIn == self.obsLength + (1 if self.difficulty else 0), f"network has {nIn} inputs, layout {self.layout} gives {self.obsLength}"
            return (True, self.difficulty, False, *self.layout[:4], self.occupied, self.landing)
        return (self.stable, nIn == 24 or self.difficulty, nIn == 25, *self.layout[:4], self.occupied, self.landing)

    def observe(self, alive, nIn):
        '''alive: int64 array of game indices. nIn: 23, 24 (difficulty) or 25 (target mode). Returns float64 [len(alive), nIn].'''
        out = np.zeros((len(alive), nIn))
        observeMany(self.F, self.I, self.PL, self.MO, alive, out, *self.observeMode(nIn))
        return out

    def startLoop(self, nIn):
        '''Start of the fused loop (see stepObserve): returns (alive, obs). alive = int64 [G] game indices, obs = float32 [G, nIn],
        row a = observation of game alive[a]; only the first n rows are valid, n = the number stepObserve returned last (G at the start).'''
        self.loopMode = self.observeMode(nIn)
        self.obs = np.zeros((self.G, nIn), dtype=np.float32)
        alive = np.arange(self.G, dtype=np.int64)
        observeFloat32(self.F, self.I, self.PL, self.MO, alive, self.G, self.obs, *self.loopMode)
        return alive, self.obs

    def stepObserve(self, alive, n, y, applied, smooth, repeat=1, decide=False):
        '''One call instead of decide + step + done-filter + observe: y = network output [n, nOut] for the games alive[:n].
        applied = float64 [G] smoothed steer per game (changed in place; smooth >= 1 = no smoothing). Drops the finished games from
        alive[:n] in place, writes the next observations to obs[:k] (the array startLoop returned) and returns k.
        decide: target mode, outputs 2..6 are the slot votes (as decide()).'''
        return stepObserveMany(self.F, self.I, self.PL, self.MO, self.BU, self.MT, alive, n, np.ascontiguousarray(y, dtype=np.float32), applied,
                               float(smooth), repeat, self.maxNoProgress, bool(decide), self.obs, *self.loopMode)

    def decide(self, alive, votes):
        decideTargets(self.F, self.I, self.PL, alive, np.ascontiguousarray(votes, dtype=np.float64))

    def step(self, alive, steer, shoot, repeat=1):
        stepMany(self.F, self.I, self.PL, self.MO, self.BU, self.MT, alive, np.ascontiguousarray(steer, dtype=np.float64),
                 np.ascontiguousarray(shoot, dtype=np.float64), repeat, self.maxNoProgress)

    def causes(self):
        '''Why each game ended: 0 running, 1 monster while rising, 2 monster otherwise, 3 fell, 4 no progress.'''
        return self.I[:, CAUSE]

    def done(self):
        return self.I[:, DONE] == 1

    def scores(self):
        return self.F[:, TOTAL] - self.F[:, OFFSET]
