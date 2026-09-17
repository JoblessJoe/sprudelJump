try:
    import pygame
except ImportError:
    pygame = None

from env import (SprudelJumpEnv, SCREEN_WIDTH, SCREEN_HEIGHT,
                 PLAYER_WIDTH, PLAYER_HEIGHT, PLATFORM_HEIGHT,
                 MONSTER_WIDTH, MONSTER_HEIGHT, BULLET_WIDTH, BULLET_HEIGHT)

FPS = 60
BG_TOP = (28, 22, 58)
BG_BOTTOM = (70, 40, 110)
PLATFORM_COLOR = (90, 200, 170)
BREAKABLE_COLOR = (235, 160, 95)
MONSTER_COLOR = (230, 70, 90)
BULLET_COLOR = (250, 240, 200)
PLAYER_COLOR = (250, 200, 60)
HUD_BG = (0, 0, 0, 120)


def makeBackground():
    bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    for y in range(SCREEN_HEIGHT):
        t = y / SCREEN_HEIGHT
        color = [int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)]
        pygame.draw.line(bg, color, (0, y), (SCREEN_WIDTH, y))
    return bg


def main():
    if pygame is None:
        print("pygame not installed; demo_render.py is optional. Use env.py directly.")
        return
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("SprudelJump")
    font = pygame.font.SysFont(None, 26)
    bigFont = pygame.font.SysFont(None, 48)
    background = makeBackground()
    env = SprudelJumpEnv()
    gameOverUntil = 0

    def sync(showGameOver):
        screen.blit(background, (0, 0))
        for x, y, w, breakable in env.platforms:
            color = BREAKABLE_COLOR if breakable else PLATFORM_COLOR
            pygame.draw.rect(screen, color, (int(x), int(y), int(w), PLATFORM_HEIGHT), border_radius=6)
        for mx, my in env.monsters:
            pygame.draw.ellipse(screen, MONSTER_COLOR, (int(mx), int(my), MONSTER_WIDTH, MONSTER_HEIGHT))
        for bx, by in env.bullets:
            pygame.draw.rect(screen, BULLET_COLOR, (int(bx), int(by), BULLET_WIDTH, BULLET_HEIGHT))
        px, py = int(env.playerX), int(env.playerY)
        pygame.draw.rect(screen, PLAYER_COLOR, (px, py, PLAYER_WIDTH, PLAYER_HEIGHT), border_radius=10)

        hud = pygame.Surface((150, 34), pygame.SRCALPHA)
        hud.fill(HUD_BG)
        screen.blit(hud, (8, 8))
        screen.blit(font.render(f"Height {round(env.totalHeight)}", True, (240, 240, 240)), (16, 15))

        if showGameOver:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 130))
            screen.blit(overlay, (0, 0))
            msg = bigFont.render("Game Over", True, (255, 255, 255))
            screen.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)))
            sub = font.render("press R to restart", True, (220, 220, 220))
            screen.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)))

        pygame.display.flip()

    sync(False)
    run = True
    clock = pygame.time.Clock()
    while run:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                gameOverUntil = 0

        if now < gameOverUntil:
            sync(True)
            clock.tick(FPS)
            continue

        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        shoot = 1.0 if keys[pygame.K_SPACE] else 0.0
        if left and right:
            steer = 0.5
        elif left:
            steer = 0.0
        elif right:
            steer = 1.0
        else:
            steer = 0.5
        s, f, done = env.step([steer, shoot])
        if done:
            gameOverUntil = now + 1200
        sync(False)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
