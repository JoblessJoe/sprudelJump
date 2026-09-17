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


class SprudelJumpEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.playerX = SCREEN_WIDTH / 2
        self.playerY = SCREEN_HEIGHT / 2
        self.velY = 0.0
        self.totalHeight = 0.0
        self.platforms = []
        y = SCREEN_HEIGHT
        while y > 0:
            self.platforms.append([random.uniform(0, SCREEN_WIDTH - PLATFORM_WIDTH), y, PLATFORM_WIDTH])
            y -= random.uniform(PLATFORM_GAP_MIN, PLATFORM_GAP_MAX)
        self.platforms.append([self.playerX - PLATFORM_WIDTH / 2, self.playerY + PLAYER_HEIGHT + 10, PLATFORM_WIDTH])
        return self._getState()

    def step(self, action):
        self.playerX += (action - 0.5) * 2 * HORIZONTAL_SPEED
        if self.playerX < -PLAYER_WIDTH:
            self.playerX = SCREEN_WIDTH
        if self.playerX > SCREEN_WIDTH:
            self.playerX = -PLAYER_WIDTH
        self.velY = min(self.velY + GRAVITY, MAX_FALL_SPEED)
        feetPrev = self.playerY + PLAYER_HEIGHT
        self.playerY += self.velY
        feet = self.playerY + PLAYER_HEIGHT
        if self.velY > 0:
            for p in self.platforms:
                if feetPrev <= p[1] <= feet and self.playerX < p[0] + p[2] and self.playerX + PLAYER_WIDTH > p[0]:
                    self.velY = BOUNCE_VELOCITY
        if self.playerY < SCROLL_THRESHOLD_Y:
            scrollAmount = SCROLL_THRESHOLD_Y - self.playerY
            self.totalHeight += scrollAmount
            self.playerY = SCROLL_THRESHOLD_Y
            for p in self.platforms:
                p[1] += scrollAmount
        self.platforms = [p for p in self.platforms if p[1] <= SCREEN_HEIGHT]
        while self.platforms and min(p[1] for p in self.platforms) > 0:
            self._spawnPlatform()
        done = self.playerY > SCREEN_HEIGHT
        return (self._getState(), self.totalHeight, done)

    def _spawnPlatform(self):
        t = min(1.0, self.totalHeight / DIFFICULTY_MAX_HEIGHT)
        gapMin = PLATFORM_GAP_MIN + t * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
        gapMax = PLATFORM_GAP_MAX + t * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
        width = PLATFORM_WIDTH - t * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
        topY = min(p[1] for p in self.platforms)
        self.platforms.append([random.uniform(0, SCREEN_WIDTH - width), topY - random.uniform(gapMin, gapMax), width])

    def render(self):
        pass

    def _getState(self):
        state = [self.playerX / SCREEN_WIDTH, max(-1.0, min(1.0, self.velY / MAX_FALL_SPEED))]
        above = sorted((p for p in self.platforms if p[1] < self.playerY), key=lambda p: self.playerY - p[1])[:5]
        for p in above:
            state.append(max(-1.0, min(1.0, (p[0] - self.playerX) / SCREEN_WIDTH)))
            state.append(max(0.0, min(1.0, (self.playerY - p[1]) / SCREEN_HEIGHT)))
        while len(state) < 12:
            state.extend([0.0, 1.0])
        return state
