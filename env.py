import random

SCREEN_WIDTH = 400
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


class SprudelJumpEnv:
    '''
    Headless Doodle Jump clone. Gym-style interface: reset() to start/restart
    an episode, step(action) to advance one frame. No rendering here — see
    demo_render.py for the human-playable pygame wrapper around this class.
    '''

    def __init__(self, maxFramesWithoutProgress=300):
        '''maxFramesWithoutProgress: episode ends once the score (totalHeight)
        hasn't increased for this many frames in a row (stops agents from
        surviving forever without climbing, e.g. bouncing on the same
        platform(s)). One full bounce is ~65 frames, so 300 is ~4-5 bounces.
        None disables the rule.'''
        self.maxFramesWithoutProgress = maxFramesWithoutProgress
        self.reset()

    def reset(self):
        '''Starts a new episode: resets player/platforms/monsters/bullets to
        their initial state and returns the first state vector (see
        _getState).'''
        self.playerX = SCREEN_WIDTH / 2
        self.playerY = SCREEN_HEIGHT / 2
        self.velY = 0.0
        self.totalHeight = 0.0
        self.platforms = []
        self.monsters = []
        self.bullets = []
        self.cooldown = 0
        self.brokenCount = 0
        self.stompCount = 0
        self.bulletKillCount = 0
        self.caughtByMonster = False
        self.bestHeight = 0.0
        self.framesWithoutProgress = 0
        y = SCREEN_HEIGHT
        while y > 0:
            self.platforms.append([random.uniform(0, SCREEN_WIDTH - PLATFORM_WIDTH), y, PLATFORM_WIDTH, False])
            y -= random.uniform(PLATFORM_GAP_MIN, PLATFORM_GAP_MAX)
        self.platforms.append([self.playerX - PLATFORM_WIDTH / 2, self.playerY + PLAYER_HEIGHT + 10, PLATFORM_WIDTH, False])
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
            by a monster, OR the score hasn't increased for
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
        if self.totalHeight > self.bestHeight:
            self.bestHeight = self.totalHeight
            self.framesWithoutProgress = 0
        else:
            self.framesWithoutProgress += 1
        stuck = self.maxFramesWithoutProgress is not None and self.framesWithoutProgress >= self.maxFramesWithoutProgress
        done = fatal or self.playerY > SCREEN_HEIGHT or stuck
        return (self._getState(), self.totalHeight, done)

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
        breakable = random.random() < (BREAKABLE_CHANCE_BASE + t * (BREAKABLE_CHANCE_MAX - BREAKABLE_CHANCE_BASE))
        self.platforms.append([random.uniform(0, SCREEN_WIDTH - width), topY - random.uniform(gapMin, gapMax), width, breakable])
        if random.random() < (MONSTER_SPAWN_CHANCE_BASE + t * (MONSTER_SPAWN_CHANCE_MAX - MONSTER_SPAWN_CHANCE_BASE)):
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
        roughly [-1, 1] or [0, 1] so it can be fed straight into a network:
          [0]    playerX, normalized by SCREEN_WIDTH
          [1]    velY, normalized by MAX_FALL_SPEED (clamped to [-1, 1])
          [2:17] 5 nearest platforms above the player, 3 floats each:
                 (relative x, relative y, is-breakable). Missing slots are
                 padded with [0.0, 1.0, 0.0] (i.e. "infinitely far below").
          [17:23] 3 nearest monsters above the player, 2 floats each:
                 (relative x, relative y). Missing slots padded [0.0, 1.0].
        Bullets are NOT included — deliberately left out of state.
        '''
        state = [self.playerX / SCREEN_WIDTH, max(-1.0, min(1.0, self.velY / MAX_FALL_SPEED))]
        above = sorted((p for p in self.platforms if p[1] < self.playerY), key=lambda p: self.playerY - p[1])[:5]
        for p in above:
            state.append(max(-1.0, min(1.0, (p[0] - self.playerX) / SCREEN_WIDTH)))
            state.append(max(0.0, min(1.0, (self.playerY - p[1]) / SCREEN_HEIGHT)))
            state.append(1.0 if p[3] else 0.0)
        while len(state) < 17:
            state.extend([0.0, 1.0, 0.0])
        aboveM = sorted((m for m in self.monsters if m[1] < self.playerY), key=lambda m: self.playerY - m[1])[:3]
        for m in aboveM:
            state.append(max(-1.0, min(1.0, (m[0] - self.playerX) / SCREEN_WIDTH)))
            state.append(max(0.0, min(1.0, (self.playerY - m[1]) / SCREEN_HEIGHT)))
        while len(state) < 23:
            state.extend([0.0, 1.0])
        return state
