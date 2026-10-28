'''
Numba-compiled copy of SprudelJumpEnv for evaluating MANY games at once (one network, many seeds, lockstep).
The Python env (env.py) stays the reference: every rule here is a line-by-line port of it, and the level generator uses
CPython's own Mersenne Twister (seeding, random(), uniform(), randint()), so the same seed gives the same levels and, with
the same actions, the same scores bit for bit (see test_fastEnv.py).

Observation: the fixed layout of env.py (3 platforms below, 2 above, 1 monster below, 2 above, landing prediction per platform; 33 floats).
Practice modes: startHeight / zeroFraction / monsterFraction. Seeds must be ints in [0, 2**32).

Typical use (see tensorNetwork/proSprudler.evaluateNetworkFast):
    b = FastBatch(seeds, maxStart, minStart, zeroFraction, monsterFraction, monsterMult)
    alive, obs = b.startLoop(); n = len(alive)
    while n:
        y = network(obs[:n])                 # float32 [n, nOut]
        n = b.stepObserve(alive, n, y, applied, smooth, repeat)     # one compiled call: smoothing, frames, drop finished games, next observation
    b.scores()                               # points beyond the start, per game
(FastBatch.run / runGroup play the whole loop with the compiled forward pass.)
'''
import numpy as np
from numba import njit
try:
    import scipy.linalg.cython_blas    # numba's np.dot calls the BLAS through scipy; without scipy forwardNet cannot be compiled
    HAVE_BLAS = True
except ImportError:
    HAVE_BLAS = False


from env import (SCREEN_WIDTH, HALF_WIDTH, SCREEN_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT, PLATFORM_WIDTH, GRAVITY, BOUNCE_VELOCITY,
                 MAX_FALL_SPEED, HORIZONTAL_SPEED, STEER_DEAD_ZONE, SCROLL_THRESHOLD_Y, PLATFORM_GAP_MIN, PLATFORM_GAP_MAX, DIFFICULTY_MAX_HEIGHT,
                 PLATFORM_GAP_MIN_HARD, PLATFORM_GAP_MAX_HARD, PLATFORM_WIDTH_MIN, BREAKABLE_CHANCE_BASE, BREAKABLE_CHANCE_MAX,
                 MONSTER_WIDTH, MONSTER_HEIGHT, MONSTER_SPAWN_CHANCE_BASE, MONSTER_SPAWN_CHANCE_MAX, MONSTER_SPAWN_CHANCE_CAP,
                 MONSTER_KILL_BONUS, BULLET_WIDTH, BULLET_HEIGHT, BULLET_SPEED, BULLET_COOLDOWN_FRAMES, PROGRESS_MARGIN,
                 PLATFORMS_BELOW, PLATFORMS_ABOVE, MONSTERS_BELOW, MONSTERS_ABOVE, OBSERVATION_LENGTH)

LAND_TMAX, LAND_TOLERANCE = 100, 40.0    # landingInfo, see env.py
MAXP, MAXM, MAXB = 48, 48, 24          # array capacities per game (a screen holds ~12 platforms, ~7 bullets)
# float state columns F[g, :]
PX, PY, VY, TOTAL, OFFSET, BEST, MMULT, TOPY = 0, 1, 2, 3, 4, 5, 6, 7    # TOPY: y of the topmost platform after the last frame
# behaviour counters (they only read the game, never change it): bullets fired, bullets fired while no monster was visible on screen, sum of the
# absolute change of the applied steer between two decisions, the applied steer of the last decision (starts at 0.5 = straight),
# network decisions the game lived through
SHOTS, SHOTS_FREE, STEER_VAR, LAST_STEER, DECISIONS, HITS = 8, 9, 10, 11, 12, 13    # HITS: monsters killed by a bullet
DETOUR, PATH_SUM, SIGNED_SUM = 14, 15, 16    # DETOUR: sideways distance travelled minus net displacement, summed over the finished jumps (see env.py)
# int state columns I[g, :]
COOL, FWP, NP, NM, NB, DONE, BOUNCED, CAUSE = 0, 1, 2, 3, 4, 5, 6, 7   # CAUSE: 0 running, 1 monster while rising, 2 monster otherwise, 3 fell, 4 no progress
# platform columns: x, y, width, breakable
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
    d = steer - 0.5
    move = 0.0
    if d > STEER_DEAD_ZONE:
        move = (d - STEER_DEAD_ZONE) / (0.5 - STEER_DEAD_ZONE) * HORIZONTAL_SPEED
    elif d < -STEER_DEAD_ZONE:
        move = (d + STEER_DEAD_ZONE) / (0.5 - STEER_DEAD_ZONE) * HORIZONTAL_SPEED
    px = F[g, PX] + move
    F[g, PATH_SUM] += abs(move)
    F[g, SIGNED_SUM] += move
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
            if not drop:
                if k != i:
                    for c in range(4):
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
        F[g, SHOTS] += 1.0
        visible = False
        for i in range(nM):
            if MO[g, i, 1] + MONSTER_HEIGHT > 0:    # the monster list is already cleaned of everything below the screen
                visible = True
                break
        if not visible:
            F[g, SHOTS_FREE] += 1.0
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
    if nB > 0 and nM > 0:
        # a bullet can hit a monster only if by < my + MONSTER_HEIGHT and by + BULLET_HEIGHT > my: with the monsters' lowest / highest y, a bullet
        # clear of that whole band cannot hit any (the same float expressions as in the test below; adding a constant is monotone), so it is skipped
        loY = hiY = MO[g, 0, 1]
        for j in range(1, nM):
            my = MO[g, j, 1]
            if my < loY:
                loY = my
            if my > hiY:
                hiY = my
        hiEdge = hiY + MONSTER_HEIGHT
        for i in range(nB):
            by = BU[g, i, 1]
            if by >= hiEdge or by + BULLET_HEIGHT <= loY:
                continue
            bx = BU[g, i, 0]
            for j in range(nM):
                if (monsterHit >> j) & 1:
                    continue
                mx = MO[g, j, 0]
                my = MO[g, j, 1]
                if my + MONSTER_HEIGHT <= 0:    # completely above the screen: cannot be hit
                    continue
                if bx < mx + MONSTER_WIDTH and bx + BULLET_WIDTH > mx and by < my + MONSTER_HEIGHT and by + BULLET_HEIGHT > my:
                    bulletGone |= 1 << i
                    monsterHit |= 1 << j
                    total += MONSTER_KILL_BONUS
                    F[g, HITS] += 1.0
                    break    # this bullet is used up (the old loop skipped its remaining monsters through the bulletGone bit)
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
                PL[g, k, 1] = y
                if y < minY:
                    minY = y
                k += 1
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
    if velY == BOUNCE_VELOCITY:    # a bounce ends a jump
        F[g, DETOUR] += F[g, PATH_SUM] - abs(F[g, SIGNED_SUM])
        F[g, PATH_SUM] = 0.0
        F[g, SIGNED_SUM] = 0.0
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
        F[g, STEER_VAR] += abs(steer[a] - F[g, LAST_STEER])
        F[g, LAST_STEER] = steer[a]
        F[g, DECISIONS] += 1.0
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
def slotIds(F, I, PL, g, out):
    '''Platform indices (into PL[g]) of the observation slots: the PLATFORMS_BELOW nearest below the feet (smallest y >= feet first), then the PLATFORMS_ABOVE nearest above (largest y < feet first); -1 = none.
    One pass over the platforms, each inserted into two small sorted lists; on equal y the lower index stays first (= the order of repeated "pick the nearest unused" scans).'''
    pb, pa = PLATFORMS_BELOW, PLATFORMS_ABOVE
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
def monsterIds(F, I, MO, g, out):
    '''Same for monsters: the MONSTERS_BELOW nearest below the feet, then the MONSTERS_ABOVE nearest above; -1 = none.'''
    mb, ma = MONSTERS_BELOW, MONSTERS_ABOVE
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
def landingFeatures(F, PL, g, row, c, i, feet, t0, K, A, Z):
    '''Landing prediction [T, r] of platform index i (-1 = empty slot), written to row[c], row[c + 1]. t0, K, A, Z: this frame's landPath.'''
    if i < 0:
        row[c] = 1.0
        row[c + 1] = 1.0
        return
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


@njit(cache=True)
def observeGame(F, I, PL, MO, g, row, ids, mids):
    '''Writes the OBSERVATION_LENGTH (33) state floats of game g into row (layout: see env.py _getState).'''
    feet = F[g, PY] + PLAYER_HEIGHT
    vy = F[g, VY] / MAX_FALL_SPEED
    row[0] = F[g, PX] / SCREEN_WIDTH
    row[1] = -1.0 if vy < -1.0 else (1.0 if vy > 1.0 else vy)
    c = 2
    slotIds(F, I, PL, g, ids)
    for s in range(PLATFORMS_BELOW + PLATFORMS_ABOVE):
        i = ids[s]
        if i < 0:
            row[c] = 0.0
            row[c + 1] = -1.0 if s < PLATFORMS_BELOW else 1.0
            row[c + 2] = 0.0
        else:
            row[c] = relXY(F, g, PL[g, i, 0], PL[g, i, 2])
            row[c + 1] = clampRel(feet, PL[g, i, 1])
            row[c + 2] = 1.0 if PL[g, i, 3] > 0.5 else 0.0
        c += 3
    monsterIds(F, I, MO, g, mids)
    for s in range(MONSTERS_BELOW + MONSTERS_ABOVE):
        i = mids[s]
        if i < 0:
            row[c] = 0.0
            row[c + 1] = -1.0 if s < MONSTERS_BELOW else 1.0
        else:
            row[c] = relXY(F, g, MO[g, i, 0], MONSTER_WIDTH)
            row[c + 1] = clampRel(feet, MO[g, i, 1])
        c += 2
    t0, K, A, Z = landPath(F[g, VY])
    for s in range(PLATFORMS_BELOW + PLATFORMS_ABOVE):
        landingFeatures(F, PL, g, row, c, ids[s], feet, t0, K, A, Z)
        c += 2


@njit(cache=True)
def observeMany(F, I, PL, MO, alive, out):
    '''Fills out[a] with the observation of game alive[a].'''
    ids = np.empty(PLATFORMS_BELOW + PLATFORMS_ABOVE, dtype=np.int64)
    mids = np.empty(MONSTERS_BELOW + MONSTERS_ABOVE, dtype=np.int64)
    for a in range(len(alive)):
        observeGame(F, I, PL, MO, alive[a], out[a], ids, mids)


@njit(cache=True)
def stepObserveMany(F, I, PL, MO, BU, MT, alive, n, y, applied, smooth, repeat, maxNoProgress, fromExp, out):
    '''Everything between two forward passes in one call: y[a] = network output of game alive[a] (a < n), or exp(-logit) with fromExp (see forwardNegated).
    Steer smoothing (smooth < 1: applied[g] += smooth * (steer - applied[g])), 'repeat' frames, then the finished games are
    dropped from alive[:n] (order kept) and the observation of the remaining ones is written to out[:k] as float32 (computed in float64 and rounded on
    the store, every column written for every game). Returns k.
    Same arithmetic as observe -> astype(float32) -> smoothing -> step -> filter, so the scores stay bit-identical.'''
    k = 0
    for a in range(n):
        g = alive[a]
        steer = np.float64(unit(y[a, 0], fromExp))
        if smooth < 1.0:
            applied[g] += smooth * (steer - applied[g])
            steer = applied[g]
        F[g, STEER_VAR] += abs(steer - F[g, LAST_STEER])
        F[g, LAST_STEER] = steer
        F[g, DECISIONS] += 1.0
        shoot = np.float64(unit(y[a, 1], fromExp))
        I[g, BOUNCED] = 0
        for _ in range(repeat):
            if stepGame(F, I, PL, MO, BU, MT, g, steer, shoot, maxNoProgress):
                I[g, DONE] = 1
                break
        if I[g, DONE] == 0:
            alive[k] = g
            k += 1
    observeFloat32(F, I, PL, MO, alive, k, out)
    return k


@njit(cache=True)
def observeFloat32(F, I, PL, MO, alive, n, out):
    '''observeMany for alive[:n] straight into the float32 array out[:n].'''
    observeMany(F, I, PL, MO, alive[:n], out)


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


@njit(cache=True)
def forwardNegated(x, ws, bs):
    '''forwardLogits, returning -logits (a negation is exact). With np.exp (numpy's own float32 exp, in place) and stepObserveMany(fromExp=True)
    this makes sigmoid(logits) = 1 / (1 + exp(-logits)) without the four numpy calls of the sigmoid: the add and the divide are plain IEEE float32
    operations, the same bits compiled or in numpy; only the exp must stay numpy's.'''
    last = len(ws) - 1
    for L in range(len(ws)):
        x = np.dot(x, ws[L]) + bs[L]
        if L < last:
            x = np.maximum(x, np.float32(0.0))
    return -x


@njit(cache=True)
def forwardGroupNegated(obs, alive, n, S, ws, bs, out):
    '''forwardNegated for a GROUP of networks in one call. The group's games sit in one FastBatch: child c owns games c*S .. (c+1)*S - 1, and
    alive[:n] keeps the games in order, so the games of one child are always one contiguous run of rows of obs. ws[L] is [k, in, out] (float32,
    the k networks' transposed weights stacked), bs[L] is [k, out]. For every run the same BLAS calls as forwardNegated make (same number of rows
    as when that child plays alone), so every output is bit for bit what the child would get alone; -logits go to out[a] (row a = game alive[a]).'''
    last = len(ws) - 1
    a = 0
    while a < n:
        c = alive[a] // S
        b = a + 1
        while b < n and alive[b] // S == c:
            b += 1
        x = obs[a:b]
        for L in range(len(ws)):
            x = np.dot(x, ws[L][c]) + bs[L][c]
            if L < last:
                x = np.maximum(x, np.float32(0.0))
        out[a:b, :] = -x
        a = b


@njit(cache=True)
def unit(e, fromExp):
    '''The network output: e itself, or 1 / (1 + e) when e = exp(-logit) (float32 arithmetic, as numpy's 1 / (1 + e) on a float32 array).'''
    return np.float32(1.0) / (np.float32(1.0) + e) if fromExp else e


class FastBatch:
    '''G games (one per seed) held in arrays, played in lockstep by the compiled functions above.'''

    def __init__(self, seeds, maxStartHeight=None, minStartHeight=0, zeroFraction=0.0, monsterFraction=0.0, monsterMult=1.0, maxFramesWithoutProgress=600):
        G = len(seeds)
        self.G = G
        self.maxNoProgress = -1 if maxFramesWithoutProgress is None else maxFramesWithoutProgress
        self.F = np.zeros((G, 17))
        self.F[:, LAST_STEER] = 0.5
        self.I = np.zeros((G, 8), dtype=np.int64)
        self.PL = np.zeros((G, MAXP, 4))
        self.MO = np.zeros((G, MAXM, 2))
        self.BU = np.zeros((G, MAXB, 2))
        self.MT = np.zeros((G, 625), dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.int64)
        assert (seeds >= 0).all() and (seeds < 2**32).all(), "seeds must be in [0, 2**32)"
        resetAll(self.F, self.I, self.PL, self.MO, self.MT, seeds, -1 if maxStartHeight is None else int(maxStartHeight), int(minStartHeight),
                 float(zeroFraction), float(monsterFraction), float(monsterMult))

    def observe(self, alive):
        '''alive: int64 array of game indices. Returns float64 [len(alive), OBSERVATION_LENGTH].'''
        out = np.zeros((len(alive), OBSERVATION_LENGTH))
        observeMany(self.F, self.I, self.PL, self.MO, alive, out)
        return out

    def startLoop(self):
        '''Start of the fused loop (see stepObserve): returns (alive, obs). alive = int64 [G] game indices, obs = float32 [G, OBSERVATION_LENGTH],
        row a = observation of game alive[a]; only the first n rows are valid, n = the number stepObserve returned last (G at the start).'''
        self.obs = np.zeros((self.G, OBSERVATION_LENGTH), dtype=np.float32)
        alive = np.arange(self.G, dtype=np.int64)
        observeFloat32(self.F, self.I, self.PL, self.MO, alive, self.G, self.obs)
        return alive, self.obs

    def stepObserve(self, alive, n, y, applied, smooth, repeat=1, fromExp=False):
        '''One call instead of step + done-filter + observe: y = network output [n, nOut] for the games alive[:n].
        applied = float64 [G] smoothed steer per game (changed in place; smooth >= 1 = no smoothing). Drops the finished games from
        alive[:n] in place, writes the next observations to obs[:k] (the array startLoop returned) and returns k.'''
        return stepObserveMany(self.F, self.I, self.PL, self.MO, self.BU, self.MT, alive, n, np.ascontiguousarray(y, dtype=np.float32), applied,
                               float(smooth), repeat, self.maxNoProgress, bool(fromExp), self.obs)

    def run(self, alive, n, wTs, bs, applied, smooth, repeat, frames, stop):
        '''The evaluation loop (forward -> step -> observe) until n == 0 or frames >= stop; returns (n, frames). Same results as
        `x = sigmoid(forwardLogits(...)); n = stepObserve(...)` per decision, with the per-decision Python work cut down: the sigmoid is
        split into forwardNegated + np.exp (in place, numpy's own float32 exp) + the finish inside stepObserveMany, and numpy's overflow
        handling is switched once for the whole call. wTs / bs: tuples of transposed float32 weights and float32 biases (see forwardLogits).'''
        obs = self.obs
        F, I, PL, MO, BU, MT, maxNoProgress = self.F, self.I, self.PL, self.MO, self.BU, self.MT, self.maxNoProgress
        smooth = float(smooth)
        with np.errstate(over="ignore"):
            while n and frames < stop:
                e = forwardNegated(obs[:n], wTs, bs)
                np.exp(e, out=e)
                n = stepObserveMany(F, I, PL, MO, BU, MT, alive, n, e, applied, smooth, repeat, maxNoProgress, True, obs)
                frames += repeat
        return n, frames

    def runGroup(self, alive, n, wsStack, bsStack, S, applied, smooth, repeat, frames, stop):
        '''run() for a batch made of k = G / S children that play the same S seeds one after another in the seed list (seeds * k): wsStack / bsStack
        are tuples of the layers' weights [k, in, out] and biases [k, out] of the k networks (float32, C order). One np.exp, one stepObserveMany
        and one loop iteration serve all k children per decision; the matmuls stay one set per child (forwardGroupNegated).'''
        obs = self.obs
        F, I, PL, MO, BU, MT, maxNoProgress = self.F, self.I, self.PL, self.MO, self.BU, self.MT, self.maxNoProgress
        smooth = float(smooth)
        out = np.empty((self.G, wsStack[-1].shape[2]), dtype=np.float32)
        with np.errstate(over="ignore"):
            while n and frames < stop:
                forwardGroupNegated(obs, alive, n, S, wsStack, bsStack, out)
                e = out[:n]
                np.exp(e, out=e)
                n = stepObserveMany(F, I, PL, MO, BU, MT, alive, n, e, applied, smooth, repeat, maxNoProgress, True, obs)
                frames += repeat
        return n, frames

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

    def decisions(self):
        '''Per game: how many network decisions it lived through (frames = decisions * action repeat, the last one may be cut short by death).'''
        return self.F[:, DECISIONS]

    def detours(self):
        '''Per game: summed detour of the finished jumps in px (sideways distance travelled minus the net displacement per jump).'''
        return self.F[:, DETOUR]

    def hits(self):
        '''Per game: monsters killed by a bullet (bullets fired - hits = bullets that missed).'''
        return self.F[:, HITS]

    def behaviour(self):
        '''Per game: (bullets fired, bullets fired with no monster visible, summed absolute steer change between decisions).'''
        return self.F[:, SHOTS], self.F[:, SHOTS_FREE], self.F[:, STEER_VAR]
