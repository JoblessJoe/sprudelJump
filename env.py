import random

SCREEN_WIDTH = 400
HALF_WIDTH = SCREEN_WIDTH / 2
SCREEN_HEIGHT = 700
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
PLATFORM_WIDTH = 60
PLATFORM_HEIGHT = 15
GRAVITY = 0.4
BOUNCE_VELOCITY = -13.0
MAX_FALL_SPEED = 15.0
HORIZONTAL_SPEED = 6.0
SCROLL_THRESHOLD_Y = SCREEN_HEIGHT * 0.4
PLATFORM_GAP_MIN = 80
PLATFORM_GAP_MAX = 130
DIFFICULTY_MAX_HEIGHT = 20000
PLATFORM_GAP_MIN_HARD = 100
PLATFORM_GAP_MAX_HARD = 160
PLATFORM_WIDTH_MIN = 40
BREAKABLE_CHANCE_BASE = 0.05
BREAKABLE_CHANCE_MAX = 0.30
MONSTER_WIDTH = 36
MONSTER_HEIGHT = 36
MONSTER_SPAWN_CHANCE_BASE = 0.0
MONSTER_SPAWN_CHANCE_MAX = 0.35
MONSTER_KILL_BONUS = 100.0
BULLET_WIDTH = 6
BULLET_HEIGHT = 14
BULLET_SPEED = 10.0
BULLET_COOLDOWN_FRAMES = 10
PROGRESS_MARGIN = 10.0  # px a new highest point must beat the old one by to count as progress


class SprudelJumpEnv:
    '''
    Headless Doodle Jump clone. Gym-style interface: reset() to start/restart
    an episode, step(action) to advance one frame. No rendering here — see
    demo_render.py for the human-playable pygame wrapper around this class.
    '''

    def __init__(self, maxFramesWithoutProgress=600, seed=None, difficultyInput=False):
        '''maxFramesWithoutProgress: episode ends once the player hasn't reached
        a new highest point (world altitude, not score) for this many frames in a
        row - stops agents from surviving forever without climbing, e.g. bouncing
        on the same platform(s). Altitude counts climbing on the lower part of the
        screen too, where the score doesn't rise because the screen doesn't scroll.
        One full bounce is ~65 frames; 600 frames = 10 s at the demo's 60 FPS.
        None disables the rule.
        difficultyInput: True = the state gets a 24th float, the current difficulty t (0 at the start,
        1 from DIFFICULTY_MAX_HEIGHT up - the same t the level generator uses). False = the classic 23-float state.
        seed: seeds this env's own random generator (platform/monster layout) at every
        reset(), so the same seed gives the same level. None = a fresh random level.'''
        self.maxFramesWithoutProgress = maxFramesWithoutProgress
        self.seed = seed
        self.difficultyInput = difficultyInput
        self.reset()

    def reset(self, maxStartHeight: int | None = None, seed: int | None = None, minStartHeight: int = 0, zeroFraction: float = 0.0):
        '''Starts a new episode: resets player/platforms/monsters/bullets to
        their initial state and returns the first state vector (see
        _getState).
        seed: if given, replaces the constructor's seed from now on. The env's own
        generator is re-created from the (new or constructor) seed, so one seed =
        one level, independent of other envs and of the global 'random' module.'''
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.playerX = SCREEN_WIDTH / 2
        self.playerY = SCREEN_HEIGHT / 2
        self.velY = 0.0
        # startHeight: practice mode. totalHeight (which drives spawn difficulty) starts there,
        # but startOffset is subtracted from the returned score, so no points are given for free.
        # zeroFraction: this share of the games (decided by the seeded rng, so the same seed = same decision) starts at 0 like a normal game.
        # With zeroFraction=0 no random number is used for it, so levels stay identical to before.
        if maxStartHeight is None or (zeroFraction > 0 and self.rng.random() < zeroFraction):
            self.startOffset = 0.0
        else:
            self.startOffset = self.rng.randint(minStartHeight, maxStartHeight)
        self.totalHeight = self.startOffset
        self.platforms = []
        self.monsters = []
        self.bullets = []
        self.cooldown = 0
        self.brokenCount = 0
        self.stompCount = 0
        self.bulletKillCount = 0
        self.caughtByMonster = False
        self.framesWithoutProgress = 0
        # first screen uses the same difficulty scaling as _spawnPlatform (no breakables/monsters here)
        t = min(1.0, self.totalHeight / DIFFICULTY_MAX_HEIGHT)
        gapMin = PLATFORM_GAP_MIN + t * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
        gapMax = PLATFORM_GAP_MAX + t * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
        width = PLATFORM_WIDTH - t * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
        y = SCREEN_HEIGHT
        while y > 0:
            self.platforms.append([self.rng.uniform(0, SCREEN_WIDTH - width), y, width, False])
            y -= self.rng.uniform(gapMin, gapMax)
        self.platforms.append([self.playerX - PLATFORM_WIDTH / 2, self.playerY + PLAYER_HEIGHT + 10, PLATFORM_WIDTH, False])
        self.bestAltitude = self.totalHeight + (SCREEN_HEIGHT - self.playerY)
        return self._getState()

    def step(self, action):
        '''
        action: [steer, shoot], a 2-element list, both floats in [0.0, 1.0] — this
        comes directly from a 2-output sigmoid neural network, so both MUST be
        exactly this range.
          - action[0] (steer): 0.0 = full left, 0.5 = no horizontal movement, 1.0 = full right
          - action[1] (shoot): > 0.5 fires a bullet this frame (subject to cooldown,
            see Monsters & shooting below). No aiming — always straight up.

        Returns (state, fitness, done):
          - state: new state vector, same format as reset()
          - fitness: the CURRENT total height climbed so far, PLUS any monster-kill
            bonuses earned so far (running total, not a per-step delta — see
            "Scoring" in step logic below). This IS the point counter/score — there
            is no separate score value.
          - done: True if the player fell off the bottom of the screen, OR was hit
            by a monster, OR hasn't reached a new highest point for
            maxFramesWithoutProgress frames (game over either way)
        '''
        # Per-step event counts for renderers/SFX. Distinct from list-length
        # diffs on self.platforms/self.monsters, which also shrink when
        # entries simply scroll off the bottom of the screen -- diffing
        # length would fire "broke"/"killed" on that routine cleanup too.
        self.brokenCount = 0
        self.stompCount = 0
        self.bulletKillCount = 0
        self.caughtByMonster = False
        steer = action[0] if isinstance(action, (list, tuple)) else action
        self.playerX += (steer - 0.5) * 2 * HORIZONTAL_SPEED
        if self.playerX < -PLAYER_WIDTH:
            self.playerX = SCREEN_WIDTH
        if self.playerX > SCREEN_WIDTH:
            self.playerX = -PLAYER_WIDTH
        self.velY = min(self.velY + GRAVITY, MAX_FALL_SPEED)
        feetPrev = self.playerY + PLAYER_HEIGHT
        self.playerY += self.velY
        feet = self.playerY + PLAYER_HEIGHT
        toRemove = []
        if self.velY > 0:
            for p in self.platforms:
                if feetPrev <= p[1] <= feet and self.playerX < p[0] + p[2] and self.playerX + PLAYER_WIDTH > p[0]:
                    self.velY = BOUNCE_VELOCITY
                    if p[3]:
                        toRemove.append(p)
            for p in toRemove:
                self.platforms.remove(p)
            self.brokenCount = len(toRemove)
        monsterGone = []
        fatal = False
        for m in self.monsters:
            if self.velY > 0 and feetPrev <= m[1] <= feet and self.playerX < m[0] + MONSTER_WIDTH and self.playerX + PLAYER_WIDTH > m[0]:
                monsterGone.append(m)
                self.totalHeight += MONSTER_KILL_BONUS
                self.velY = BOUNCE_VELOCITY
            elif (self.playerX < m[0] + MONSTER_WIDTH and self.playerX + PLAYER_WIDTH > m[0] and self.playerY < m[1] + MONSTER_HEIGHT and self.playerY + PLAYER_HEIGHT > m[1]):
                fatal = True
                self.caughtByMonster = True
        for m in monsterGone:
            self.monsters.remove(m)
        self.stompCount = len(monsterGone)
        shoot = action[1] if isinstance(action, (list, tuple)) else 0.0
        if shoot > 0.5 and self.cooldown == 0:
            self.bullets.append([self.playerX + PLAYER_WIDTH / 2 - BULLET_WIDTH / 2, self.playerY])
            self.cooldown = BULLET_COOLDOWN_FRAMES
        self.cooldown = max(0, self.cooldown - 1)
        for b in self.bullets:
            b[1] -= BULLET_SPEED
        self.bullets = [b for b in self.bullets if b[1] >= -BULLET_HEIGHT]
        bulletsGone, monstersHit = [], []
        for b in self.bullets:
            for m in self.monsters:
                if b in bulletsGone or m in monstersHit:
                    continue
                if b[0] < m[0] + MONSTER_WIDTH and b[0] + BULLET_WIDTH > m[0] and b[1] < m[1] + MONSTER_HEIGHT and b[1] + BULLET_HEIGHT > m[1]:
                    bulletsGone.append(b)
                    monstersHit.append(m)
                    self.totalHeight += MONSTER_KILL_BONUS
        for b in bulletsGone:
            self.bullets.remove(b)
        for m in monstersHit:
            self.monsters.remove(m)
        self.bulletKillCount = len(monstersHit)
        if self.playerY < SCROLL_THRESHOLD_Y:
            scrollAmount = SCROLL_THRESHOLD_Y - self.playerY
            self.totalHeight += scrollAmount
            self.playerY = SCROLL_THRESHOLD_Y
            for p in self.platforms:
                p[1] += scrollAmount
            for m in self.monsters:
                m[1] += scrollAmount
            for b in self.bullets:
                b[1] += scrollAmount
        self.platforms = [p for p in self.platforms if p[1] <= SCREEN_HEIGHT]
        self.monsters = [m for m in self.monsters if m[1] <= SCREEN_HEIGHT]
        self.bullets = [b for b in self.bullets if b[1] <= SCREEN_HEIGHT]
        while self.platforms and min(p[1] for p in self.platforms) > 0:
            self._spawnPlatform()
        # world altitude of the player: rises when moving up the screen AND when the screen scrolls
        altitude = self.totalHeight + (SCREEN_HEIGHT - self.playerY)
        if altitude > self.bestAltitude + PROGRESS_MARGIN:  # margin: repeated bounces on one platform peak at
            self.bestAltitude = altitude                     # almost the same height and must not count as progress
            self.framesWithoutProgress = 0
        else:
            self.framesWithoutProgress += 1
        stuck = self.maxFramesWithoutProgress is not None and self.framesWithoutProgress >= self.maxFramesWithoutProgress
        done = fatal or self.playerY > SCREEN_HEIGHT or stuck
        return (self._getState(), self.totalHeight - self.startOffset, done)

    def _spawnPlatform(self):
        '''Adds one new platform above the current highest one, and rolls a
        chance to put a monster on it. Gap size, platform width, breakable
        chance, and monster chance all scale with difficulty t (0 at
        totalHeight=0, 1 at DIFFICULTY_MAX_HEIGHT and beyond).'''
        t = min(1.0, self.totalHeight / DIFFICULTY_MAX_HEIGHT)
        gapMin = PLATFORM_GAP_MIN + t * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
        gapMax = PLATFORM_GAP_MAX + t * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
        width = PLATFORM_WIDTH - t * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
        topY = min(p[1] for p in self.platforms)
        breakable = self.rng.random() < (BREAKABLE_CHANCE_BASE + t * (BREAKABLE_CHANCE_MAX - BREAKABLE_CHANCE_BASE))
        self.platforms.append([self.rng.uniform(0, SCREEN_WIDTH - width), topY - self.rng.uniform(gapMin, gapMax), width, breakable])
        if self.rng.random() < (MONSTER_SPAWN_CHANCE_BASE + t * (MONSTER_SPAWN_CHANCE_MAX - MONSTER_SPAWN_CHANCE_BASE)):
            px, py, pw = self.platforms[-1][0], self.platforms[-1][1], self.platforms[-1][2]
            self.monsters.append([px + pw / 2 - MONSTER_WIDTH / 2, py - MONSTER_HEIGHT])

    def render(self):
        '''No-op. This class is headless by design — visuals live entirely in
        demo_render.py, which reads player/platforms/monsters/bullets directly
        and never calls this.'''
        pass

    def _getState(self):
        '''
        Builds the 23-float observation vector, all values pre-normalized to
        roughly [-1, 1] or [0, 1] so it can be fed straight into a network.
        Positions are relative to the player's FEET (what lands on platforms)
        and horizontal CENTER:
          [0]    playerX, normalized by SCREEN_WIDTH
          [1]    velY, normalized by MAX_FALL_SPEED (clamped to [-1, 1])
          [2:17] 5 platforms nearest to the feet, ABOVE OR BELOW, 3 floats each:
                 (relative x of platform center, wrap-aware, signed relative y, is-breakable).
                 relative y > 0 = platform above the feet, < 0 = below.
                 Platforms below matter: you land on them while falling.
                 Missing slots are padded with [0.0, -1.0, 0.0] ("far below").
          [17:23] 3 monsters nearest to the feet, above or below, 2 floats each:
                 (relative x of monster center, signed relative y). Missing slots
                 padded [0.0, -1.0].
        Sorted nearest-first. Bullets are NOT included - deliberately left out of state.
        Relative x accounts for the horizontal screen wrap: it is the SHORTER way
        around (through the edge if that is closer), so it lies in [-0.5, 0.5].
        '''
        # Hot path (runs every frame): clamps are inlined as conditionals instead of
        # min/max/helper calls - same values, roughly half the cost.
        feet = self.playerY + PLAYER_HEIGHT
        centerX = self.playerX + PLAYER_WIDTH / 2
        vy = self.velY / MAX_FALL_SPEED
        state = [self.playerX / SCREEN_WIDTH, -1.0 if vy < -1.0 else (1.0 if vy > 1.0 else vy)]
        for p in sorted(self.platforms, key=lambda p: abs(feet - p[1]))[:5]:
            dx = p[0] + p[2] / 2 - centerX
            if dx > HALF_WIDTH:  # shorter way around is through the screen edge
                dx -= SCREEN_WIDTH
            elif dx < -HALF_WIDTH:
                dx += SCREEN_WIDTH
            rx = dx / SCREEN_WIDTH
            ry = (feet - p[1]) / SCREEN_HEIGHT
            state.append(rx)
            state.append(-1.0 if ry < -1.0 else (1.0 if ry > 1.0 else ry))
            state.append(1.0 if p[3] else 0.0)
        while len(state) < 17:
            state.extend([0.0, -1.0, 0.0])
        for m in sorted(self.monsters, key=lambda m: abs(feet - m[1]))[:3]:
            dx = m[0] + MONSTER_WIDTH / 2 - centerX
            if dx > HALF_WIDTH:
                dx -= SCREEN_WIDTH
            elif dx < -HALF_WIDTH:
                dx += SCREEN_WIDTH
            rx = dx / SCREEN_WIDTH
            ry = (feet - m[1]) / SCREEN_HEIGHT
            state.append(rx)
            state.append(-1.0 if ry < -1.0 else (1.0 if ry > 1.0 else ry))
        while len(state) < 23:
            state.extend([0.0, -1.0])
        if self.difficultyInput:
            t = self.totalHeight / DIFFICULTY_MAX_HEIGHT
            state.append(1.0 if t > 1.0 else t)
        return state
