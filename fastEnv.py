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
'''
import numpy as np
from numba import njit

from env import (SCREEN_WIDTH, HALF_WIDTH, SCREEN_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT, PLATFORM_WIDTH, GRAVITY, BOUNCE_VELOCITY,
                 MAX_FALL_SPEED, HORIZONTAL_SPEED, SCROLL_THRESHOLD_Y, PLATFORM_GAP_MIN, PLATFORM_GAP_MAX, DIFFICULTY_MAX_HEIGHT,
                 PLATFORM_GAP_MIN_HARD, PLATFORM_GAP_MAX_HARD, PLATFORM_WIDTH_MIN, BREAKABLE_CHANCE_BASE, BREAKABLE_CHANCE_MAX,
                 MONSTER_WIDTH, MONSTER_HEIGHT, MONSTER_SPAWN_CHANCE_BASE, MONSTER_SPAWN_CHANCE_MAX, MONSTER_SPAWN_CHANCE_CAP,
                 MONSTER_KILL_BONUS, BULLET_WIDTH, BULLET_HEIGHT, BULLET_SPEED, BULLET_COOLDOWN_FRAMES, PROGRESS_MARGIN)

MAXP, MAXM, MAXB = 48, 48, 24          # array capacities per game (a screen holds ~12 platforms, ~7 bullets)
# float state columns F[g, :]
PX, PY, VY, TOTAL, OFFSET, BEST, MMULT = 0, 1, 2, 3, 4, 5, 6
# 'ghost' of a removed target platform (the Python env keeps feeding the frozen last position of the platform object for one frame)
GHOST_FLAG, GHOST_X, GHOST_Y, GHOST_W = 8, 9, 10, 11
# int state columns I[g, :]
COOL, FWP, NP, NM, NB, DONE, BOUNCED, NEXTID, TARGET = 0, 1, 2, 3, 4, 5, 6, 7, 8
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
        for kk in range(624):
            y = (mt[kk] & 0x80000000) | (mt[(kk + 1) % 624] & 0x7FFFFFFF)
            v = mt[(kk + 397) % 624] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908B0DF
            mt[kk] = v
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
    t = min(1.0, F[g, TOTAL] / DIFFICULTY_MAX_HEIGHT)
    gapMin = PLATFORM_GAP_MIN + t * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
    gapMax = PLATFORM_GAP_MAX + t * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
    width = PLATFORM_WIDTH - t * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
    topY = PL[g, 0, 1]
    for i in range(1, I[g, NP]):
        if PL[g, i, 1] < topY:
            topY = PL[g, i, 1]
    breakable = 1.0 if rndRandom(MT[g]) < (BREAKABLE_CHANCE_BASE + t * (BREAKABLE_CHANCE_MAX - BREAKABLE_CHANCE_BASE)) else 0.0
    x = rndUniform(MT[g], 0.0, SCREEN_WIDTH - width)
    y = topY - rndUniform(MT[g], gapMin, gapMax)
    addPlatform(PL, I, g, x, y, width, breakable)
    if rndRandom(MT[g]) < min(MONSTER_SPAWN_CHANCE_CAP, (MONSTER_SPAWN_CHANCE_BASE + t * (MONSTER_SPAWN_CHANCE_MAX - MONSTER_SPAWN_CHANCE_BASE)) * F[g, MMULT]):
        n = I[g, NM]
        MO[g, n, 0] = x + width / 2 - MONSTER_WIDTH / 2
        MO[g, n, 1] = y - MONSTER_HEIGHT
        I[g, NM] = n + 1


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
    if velY > 0:
        removed = False
        keep = np.ones(nP, dtype=np.bool_)
        for i in range(nP):
            py = PL[g, i, 1]
            if feetPrev <= py and py <= feet and px < PL[g, i, 0] + PL[g, i, 2] and px + PLAYER_WIDTH > PL[g, i, 0]:
                velY = BOUNCE_VELOCITY
                I[g, BOUNCED] = 1
                if PL[g, i, 3] > 0.5:
                    keep[i] = False
                    removed = True
                    if PL[g, i, 4] == I[g, TARGET]:
                        F[g, GHOST_FLAG] = 1.0
                        F[g, GHOST_X] = PL[g, i, 0]
                        F[g, GHOST_Y] = PL[g, i, 1]
                        F[g, GHOST_W] = PL[g, i, 2]
        if removed:
            k = 0
            for i in range(nP):
                if keep[i]:
                    if k != i:
                        for c in range(5):
                            PL[g, k, c] = PL[g, i, c]
                    k += 1
            nP = k
            I[g, NP] = k
    nM = I[g, NM]
    fatal = False
    goneAny = False
    keepM = np.ones(nM, dtype=np.bool_)
    total = F[g, TOTAL]
    for i in range(nM):
        mx = MO[g, i, 0]
        my = MO[g, i, 1]
        overlapX = px < mx + MONSTER_WIDTH and px + PLAYER_WIDTH > mx
        if velY > 0 and feetPrev <= my and my <= feet and overlapX:
            keepM[i] = False
            goneAny = True
            total += MONSTER_KILL_BONUS
            velY = BOUNCE_VELOCITY
            I[g, BOUNCED] = 1
        elif overlapX and playerY < my + MONSTER_HEIGHT and playerY + PLAYER_HEIGHT > my:
            fatal = True
    if goneAny:
        k = 0
        for i in range(nM):
            if keepM[i]:
                if k != i:
                    MO[g, k, 0] = MO[g, i, 0]
                    MO[g, k, 1] = MO[g, i, 1]
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
    bulletGone = np.zeros(nB, dtype=np.bool_)
    monsterHit = np.zeros(nM, dtype=np.bool_)
    anyHit = False
    for i in range(nB):
        bx = BU[g, i, 0]
        by = BU[g, i, 1]
        for j in range(nM):
            if bulletGone[i] or monsterHit[j]:
                continue
            mx = MO[g, j, 0]
            my = MO[g, j, 1]
            if bx < mx + MONSTER_WIDTH and bx + BULLET_WIDTH > mx and by < my + MONSTER_HEIGHT and by + BULLET_HEIGHT > my:
                bulletGone[i] = True
                monsterHit[j] = True
                anyHit = True
                total += MONSTER_KILL_BONUS
    if anyHit:
        k = 0
        for i in range(nB):
            if not bulletGone[i]:
                BU[g, k, 0] = BU[g, i, 0]
                BU[g, k, 1] = BU[g, i, 1]
                k += 1
        nB = k
        k = 0
        for j in range(nM):
            if not monsterHit[j]:
                MO[g, k, 0] = MO[g, j, 0]
                MO[g, k, 1] = MO[g, j, 1]
                k += 1
        nM = k
    # scrolling
    if playerY < SCROLL_THRESHOLD_Y:
        scroll = SCROLL_THRESHOLD_Y - playerY
        total += scroll
        playerY = SCROLL_THRESHOLD_Y
        for i in range(nP):
            PL[g, i, 1] += scroll
        for i in range(nM):
            MO[g, i, 1] += scroll
        for i in range(nB):
            BU[g, i, 1] += scroll
    # drop what fell off the bottom (order preserved)
    k = 0
    for i in range(nP):
        if PL[g, i, 1] <= SCREEN_HEIGHT:
            if k != i:
                for c in range(5):
                    PL[g, k, c] = PL[g, i, c]
            k += 1
        elif PL[g, i, 4] == I[g, TARGET]:
            F[g, GHOST_FLAG] = 1.0
            F[g, GHOST_X] = PL[g, i, 0]
            F[g, GHOST_Y] = PL[g, i, 1]
            F[g, GHOST_W] = PL[g, i, 2]
    nP = k
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
    # new platforms at the top
    while I[g, NP] > 0:
        minY = PL[g, 0, 1]
        for i in range(1, I[g, NP]):
            if PL[g, i, 1] < minY:
                minY = PL[g, i, 1]
        if not minY > 0:
            break
        spawnPlatform(F, I, PL, MO, MT, g)
    altitude = F[g, TOTAL] + (SCREEN_HEIGHT - F[g, PY])
    if altitude > F[g, BEST] + PROGRESS_MARGIN:
        F[g, BEST] = altitude
        I[g, FWP] = 0
    else:
        I[g, FWP] += 1
    stuck = maxNoProgress >= 0 and I[g, FWP] >= maxNoProgress
    return fatal or F[g, PY] > SCREEN_HEIGHT or stuck


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
def slotIds(F, I, PL, g, out):
    '''Platform indices (into PL[g]) of the 5 stableSlots: 3 nearest below the feet (smallest y >= feet first), 2 nearest above (largest y < feet first); -1 = none.'''
    feet = F[g, PY] + PLAYER_HEIGHT
    nP = I[g, NP]
    for s in range(5):
        out[s] = -1
    for s in range(3):
        best = -1
        for i in range(nP):
            y = PL[g, i, 1]
            if y >= feet:
                already = False
                for q in range(s):
                    if out[q] == i:
                        already = True
                if already:
                    continue
                if best < 0 or y < PL[g, best, 1]:
                    best = i
        out[s] = best
    for s in range(3, 5):
        best = -1
        for i in range(nP):
            y = PL[g, i, 1]
            if y < feet:
                already = False
                for q in range(3, s):
                    if out[q] == i:
                        already = True
                if already:
                    continue
                if best < 0 or y > PL[g, best, 1]:
                    best = i
        out[s] = best


@njit(cache=True)
def observeGame(F, I, PL, MO, g, row, stable, difficulty):
    '''Writes the 23 (or 24 with difficulty) state floats of game g into row; returns the number written.'''
    feet = F[g, PY] + PLAYER_HEIGHT
    vy = F[g, VY] / MAX_FALL_SPEED
    row[0] = F[g, PX] / SCREEN_WIDTH
    row[1] = -1.0 if vy < -1.0 else (1.0 if vy > 1.0 else vy)
    nP = I[g, NP]
    nM = I[g, NM]
    c = 2
    if stable:
        ids = np.empty(5, dtype=np.int64)
        slotIds(F, I, PL, g, ids)
        for s in range(5):
            i = ids[s]
            if i < 0:
                row[c] = 0.0
                row[c + 1] = -1.0 if s < 3 else 1.0
                row[c + 2] = 0.0
            else:
                row[c] = relXY(F, g, PL[g, i, 0], PL[g, i, 2])
                row[c + 1] = clampRel(feet, PL[g, i, 1])
                row[c + 2] = 1.0 if PL[g, i, 3] > 0.5 else 0.0
            c += 3
        # monsters: 1 nearest below (smallest y >= feet), then the 2 nearest above (largest y < feet)
        below = -1
        for i in range(nM):
            y = MO[g, i, 1]
            if y >= feet and (below < 0 or y < MO[g, below, 1]):
                below = i
        if below < 0:
            row[c] = 0.0
            row[c + 1] = -1.0
        else:
            row[c] = relXY(F, g, MO[g, below, 0], MONSTER_WIDTH)
            row[c + 1] = clampRel(feet, MO[g, below, 1])
        c += 2
        first = -1
        for i in range(nM):
            y = MO[g, i, 1]
            if y < feet and (first < 0 or y > MO[g, first, 1]):
                first = i
        second = -1
        for i in range(nM):
            y = MO[g, i, 1]
            if y < feet and i != first and (second < 0 or y > MO[g, second, 1]):
                second = i
        for i in (first, second):
            if i < 0:
                row[c] = 0.0
                row[c + 1] = 1.0
            else:
                row[c] = relXY(F, g, MO[g, i, 0], MONSTER_WIDTH)
                row[c + 1] = clampRel(feet, MO[g, i, 1])
            c += 2
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
def observeMany(F, I, PL, MO, alive, out, stable, difficulty, targetMode):
    '''Fills out[a] with the observation of game alive[a]. In target mode (out has 25 columns) the last two are the committed
    target's relative x / signed relative y ([0, -1] without a target).'''
    for a in range(len(alive)):
        g = alive[a]
        c = observeGame(F, I, PL, MO, g, out[a], stable, difficulty)
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
        slotIds(F, I, PL, g, ids)
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


class FastBatch:
    '''G games (one per seed) held in arrays, played in lockstep by the compiled functions above.'''

    def __init__(self, seeds, maxStartHeight=None, minStartHeight=0, zeroFraction=0.0, monsterFraction=0.0, monsterMult=1.0,
                 stable=False, difficulty=False, maxFramesWithoutProgress=600):
        G = len(seeds)
        self.G = G
        self.stable, self.difficulty = stable, difficulty
        self.maxNoProgress = -1 if maxFramesWithoutProgress is None else maxFramesWithoutProgress
        self.F = np.zeros((G, 12))
        self.I = np.zeros((G, 9), dtype=np.int64)
        self.PL = np.zeros((G, MAXP, 5))
        self.MO = np.zeros((G, MAXM, 2))
        self.BU = np.zeros((G, MAXB, 2))
        self.MT = np.zeros((G, 625), dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.int64)
        assert (seeds >= 0).all() and (seeds < 2**32).all(), "seeds must be in [0, 2**32)"
        resetAll(self.F, self.I, self.PL, self.MO, self.MT, seeds, -1 if maxStartHeight is None else int(maxStartHeight), int(minStartHeight),
                 float(zeroFraction), float(monsterFraction), float(monsterMult))

    def observe(self, alive, nIn):
        '''alive: int64 array of game indices. nIn: 23, 24 (difficulty) or 25 (target mode). Returns float64 [len(alive), nIn].'''
        out = np.zeros((len(alive), nIn))
        observeMany(self.F, self.I, self.PL, self.MO, alive, out, self.stable, nIn == 24 or self.difficulty, nIn == 25)
        return out

    def decide(self, alive, votes):
        decideTargets(self.F, self.I, self.PL, alive, np.ascontiguousarray(votes, dtype=np.float64))

    def step(self, alive, steer, shoot, repeat=1):
        stepMany(self.F, self.I, self.PL, self.MO, self.BU, self.MT, alive, np.ascontiguousarray(steer, dtype=np.float64),
                 np.ascontiguousarray(shoot, dtype=np.float64), repeat, self.maxNoProgress)

    def done(self):
        return self.I[:, DONE] == 1

    def scores(self):
        return self.F[:, TOTAL] - self.F[:, OFFSET]
