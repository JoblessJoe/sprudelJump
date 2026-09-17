try:
    import pygame
except ImportError:
    pygame = None

from env import (SprudelJumpEnv, SCREEN_WIDTH, SCREEN_HEIGHT,
                 PLAYER_WIDTH, PLAYER_HEIGHT, PLATFORM_HEIGHT)

FPS = 60

def main():
    if pygame is None:
        print("pygame not installed; demo_render.py is optional. Use env.py directly.")
        return
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("SprudelJump demo")
    font = pygame.font.SysFont(None, 24)
    env = SprudelJumpEnv()

    def sync():
        screen.fill((24, 24, 34))
        for x, y, w in env.platforms:
            pygame.draw.rect(screen, (70, 130, 180), (int(x), int(y), int(w), PLATFORM_HEIGHT))
        pygame.draw.rect(screen, (240, 200, 60), (int(env.playerX), int(env.playerY), PLAYER_WIDTH, PLAYER_HEIGHT))
        screen.blit(font.render("H " + str(round(env.totalHeight, 1)), True, (230, 230, 230)), (8, 8))
        pygame.display.flip()

    sync()
    run = True
    clock = pygame.time.Clock()
    while run:
        step_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]:
            step_action = 0.5
        elif keys[pygame.K_LEFT]:
            step_action = 0.0
        elif keys[pygame.K_RIGHT]:
            step_action = 1.0
        else:
            step_action = 0.5
        s, f, done = env.step(step_action)
        if done:
            env.reset()
        sync()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
