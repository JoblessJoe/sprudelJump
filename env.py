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
    def __init__(self):
        self.reset()

    def reset(self):
        self.playerX = SCREEN_WIDTH / 2
        self.playerY = SCREEN_HEIGHT / 2
        self.velY = 0.0
        self.totalHeight = 0.0
        self.platforms = []
        self.monsters = []
        self.bullets = []
        self.cooldown = 0
        y = SCREEN_HEIGHT
        while y > 0:
            self.platforms.append([random.uniform(0, SCREEN_WIDTH - PLATFORM_WIDTH), y, PLATFORM_WIDTH, False])
            y -= random.uniform(PLATFORM_GAP_MIN, PLATFORM_GAP_MAX)
        self.platforms.append([self.playerX - PLATFORM_WIDTH / 2, self.playerY + PLAYER_HEIGHT + 10, PLATFORM_WIDTH, False])
        return self._getState()

    def step(self, action):
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
        monsterGone = []
        fatal = False
        for m in self.monsters:
            if self.velY > 0 and feetPrev <= m[1] <= feet and self.playerX < m[0] + MONSTER_WIDTH and self.playerX + PLAYER_WIDTH > m[0]:
                monsterGone.append(m)
                self.totalHeight += MONSTER_KILL_BONUS
                self.velY = BOUNCE_VELOCITY
            elif (self.playerX < m[0] + MONSTER_WIDTH and self.playerX + PLAYER_WIDTH > m[0] and self.playerY < m[1] + MONSTER_HEIGHT and self.playerY + PLAYER_HEIGHT > m[1]):
                fatal = True
        for m in monsterGone:
            self.monsters.remove(m)
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
        done = fatal or self.playerY > SCREEN_HEIGHT
        return (self._getState(), self.totalHeight, done)

    def _spawnPlatform(self):
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
        pass

    def _getState(self):
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
