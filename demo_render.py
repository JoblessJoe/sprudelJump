"""SprudelJump demo renderer: polished pixel-art UI over the headless env."""
try:
    import pygame
except ImportError:
    pygame = None

import argparse
import os
import random

import art
import sounds
from env import (SprudelJumpEnv, SCREEN_WIDTH, SCREEN_HEIGHT,
                 PLAYER_WIDTH, PLAYER_HEIGHT, PLATFORM_HEIGHT,
                 BULLET_WIDTH, BULLET_HEIGHT, BULLET_COOLDOWN_FRAMES, OBSERVATION_LENGTH)

FPS = 60
SPEEDS = (1, 2, 4, 8, 16)    # steps of the speed button (network play)
BG_TOP = (30, 22, 66)
BG_BOTTOM = (84, 46, 124)
HILL_FAR = (44, 33, 84)
HILL_NEAR = (56, 41, 100)
PLAT_TOP = (150, 230, 205)
PLAT_BODY = (82, 188, 158)
PLAT_BREAK_TOP = (255, 205, 130)
PLAT_BREAK_BODY = (222, 148, 74)
PLAT_CRACK = (120, 74, 30)
BULLET_CORE = (255, 244, 190)
BULLET_GLOW = (255, 220, 110)
HUD_BG = (12, 8, 30, 150)
HUD_TEXT = (245, 240, 255)
DIM_TEXT = (200, 192, 225)
PANEL_BG = (12, 8, 30, 215)
PANEL_BORDER = (125, 95, 205, 255)


def make_background():
    bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    for y in range(SCREEN_HEIGHT):
        t = y / SCREEN_HEIGHT
        c = [int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)]
        pygame.draw.line(bg, c, (0, y), (SCREEN_WIDTH, y))
    rng = random.Random(20240707)
    for _ in range(50):
        x, y = rng.randrange(SCREEN_WIDTH), rng.randrange(SCREEN_HEIGHT - 150)
        r = rng.choice((1, 1, 1, 2))
        star = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(star, (235, 235, 255, rng.choice((100, 140, 190))), (r, r), r)
        bg.blit(star, (x, y))
    for x in (-40, 90, 210, 320):
        pygame.draw.ellipse(bg, HILL_FAR, (x, SCREEN_HEIGHT - 150, 170, 140))
    for x in (-70, 80, 250):
        pygame.draw.ellipse(bg, HILL_NEAR, (x, SCREEN_HEIGHT - 95, 200, 100))
    return bg


def make_bullet():
    w, h = BULLET_WIDTH + 8, BULLET_HEIGHT + 8
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, BULLET_GLOW + (70,), s.get_rect(), border_radius=w // 2)
    pygame.draw.rect(s, BULLET_CORE, s.get_rect().inflate(-8, -6), border_radius=3)
    return s, w, h


def pill(font, text, text_color=HUD_TEXT, fill=HUD_BG):
    img = font.render(text, True, text_color)
    s = pygame.Surface((img.get_width() + 20, img.get_height() + 12), pygame.SRCALPHA)
    pygame.draw.rect(s, fill, s.get_rect(), border_radius=s.get_height() // 2)
    s.blit(img, (10, 6))
    return s


def draw_platform(surface, p):
    x, y, w, breakable = p
    top, body = (PLAT_BREAK_TOP, PLAT_BREAK_BODY) if breakable else (PLAT_TOP, PLAT_BODY)
    r = (int(x), int(y), int(w), PLATFORM_HEIGHT)
    pygame.draw.rect(surface, body, r, border_radius=7)
    pygame.draw.rect(surface, top, (r[0], r[1], r[2], 6), border_radius=3)
    if breakable:
        for fx in (0.25, 0.55, 0.8):
            cx = int(x + w * fx)
            pygame.draw.line(surface, PLAT_CRACK, (cx, y + 3), (cx + 3, y + 8), 2)
            pygame.draw.line(surface, PLAT_CRACK, (cx + 3, y + 8), (cx, y + 12), 1)


def load_model_policy(path, smoothing=0.96875):
    '''
    Loads a network saved by tensorNetwork's saveNetwork() (a dict with 'layers':
    list of (weights [out,in], bias [out]) tensors) and returns policy(state) ->
    (steer, shoot output). Same math as tensorNetwork's Network.forwardPass: ReLU hidden
    layers, sigmoid output. Kept inline so this repo doesn't import tensorNetwork.
    Steer smoothing like in tensorNetwork's evaluateNetwork: applied steer += smoothing * (output - applied).
    policy.reset() = new game.
    '''
    import torch
    layers = torch.load(path, map_location="cpu", weights_only=True)["layers"]  # file holds only tensors + a float
    assert layers[0][0].shape[1] == OBSERVATION_LENGTH, f"the network has {layers[0][0].shape[1]} inputs, the game gives {OBSERVATION_LENGTH}"
    mem = {"applied": 0.5}

    def reset():
        mem["applied"] = 0.5

    def policy(state):
        x = torch.tensor(state).unsqueeze(0)
        for w, b in layers[:-1]:
            x = torch.relu(x @ w.T + b)
        w, b = layers[-1]
        out = torch.sigmoid(x @ w.T + b).squeeze(0).tolist()
        steer = out[0]
        if smoothing < 1.0:
            mem["applied"] += smoothing * (steer - mem["applied"])
            steer = mem["applied"]
        return steer, out[1]
    policy.reset = reset
    return policy


def main(modelPath=None, smoothing=0.96875, repeat=5, speed=1):
    if pygame is None:
        print("pygame not installed; demo_render.py is optional. Use env.py directly.")
        return
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("SprudelJump")
    font = pygame.font.SysFont("DejaVuSans-Bold, Verdana, Arial", 20)
    bigFont = pygame.font.SysFont("DejaVuSans-Bold, Verdana, Arial", 42)
    smallFont = pygame.font.SysFont("DejaVuSans, Verdana, Arial", 16)
    background = make_background()
    playerSurf = art.player_sprite()
    monsterSurfs = art.monster_sprites()
    bulletSurf, bw, bh = make_bullet()
    try:
        pygame.display.set_icon(playerSurf)
    except pygame.error:
        pass

    policy = load_model_policy(modelPath, smoothing) if modelPath else None
    modelName = os.path.basename(modelPath) if modelPath else None
    # human play: no stuck-rule. Network play: same rule as in training, or a bouncer would loop forever.
    env = SprudelJumpEnv() if policy else SprudelJumpEnv(maxFramesWithoutProgress=None)
    state = env._getState()
    autoRestart = False
    best = 0
    newBest = False
    gameOverUntil = 0
    spaceWas = False
    sndBank = sounds.load()
    prevVelY = env.velY
    frameInRepeat = 0          # network play: the net decides on frame 0 of every 'repeat' frames, its action is held in between
    held = (0.5, 0.0)

    def sfx(name):
        if name in sndBank:
            sndBank[name].play()

    pendingPings = set()

    def ping(name):    # sound effects of single frames: collected and played once per drawn picture, so a sped-up demo does not stack the same sound many times
        pendingPings.add(name)

    def speed_button():
        img = pill(font, f"SPEED x{speed}")
        return img, img.get_rect(midbottom=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 34))


    def draw_world():
        screen.blit(background, (0, 0))
        for x, y, w, breakable in env.platforms:
            draw_platform(screen, (x, y, w, breakable))
        for mx, my in env.monsters:    # the look of a monster comes from its (never changing) x, not from its place in the list: when one dies the others keep their look
            screen.blit(monsterSurfs[int(mx) % 2], (int(mx), int(my)))
        for bx, by in env.bullets:
            screen.blit(bulletSurf, (int(bx) - (bw - BULLET_WIDTH) // 2,
                                     int(by) - (bh - BULLET_HEIGHT) // 2))
        screen.blit(playerSurf, (int(env.playerX) + (PLAYER_WIDTH - playerSurf.get_width()) // 2,
                                 int(env.playerY) + (PLAYER_HEIGHT - playerSurf.get_height()) // 2))

    def draw_hud():
        h = pill(font, f"H {max(0, round(env.totalHeight)):4d}")
        b = pill(font, f"BEST {best:4d}")
        screen.blit(h, (8, 8))
        screen.blit(b, (SCREEN_WIDTH - b.get_width() - 8, 8))
        if policy:
            tip = smallFont.render(f"AI: {modelName}     R restart   Esc quit", True, DIM_TEXT)
            screen.blit(tip, tip.get_rect(midbottom=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 10)))
            img, rect = speed_button()
            screen.blit(img, rect)
        elif env.totalHeight < 50 and best == 0:
            tip = smallFont.render("arrows / A D to steer     SPACE to shoot", True, DIM_TEXT)
            screen.blit(tip, tip.get_rect(midbottom=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 10)))

    def draw_game_over():
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 5, 20, 160))
        screen.blit(overlay, (0, 0))
        panel = pygame.Surface((290, 170), pygame.SRCALPHA)
        pygame.draw.rect(panel, PANEL_BG, panel.get_rect(), border_radius=16)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), width=2, border_radius=16)
        screen.blit(panel, panel.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        title = bigFont.render("GAME OVER", True, (255, 255, 255))
        extra = "   NEW BEST!" if newBest else f"   Best {best}"
        score = font.render(f"Height {round(env.totalHeight)}{extra}", True, HUD_TEXT)
        again = smallFont.render("R restart      Esc quit", True, DIM_TEXT)
        for img, dy in ((title, -48), (score, 4), (again, 44)):
            screen.blit(img, img.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + dy)))

    run = True
    clock = pygame.time.Clock()
    while run:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False
            if (policy and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                    and speed_button()[1].collidepoint(event.pos)):    # speed button: next step of SPEEDS, back to x1 after the last
                speed = next((s for s in SPEEDS if s > speed), SPEEDS[0])
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = env.reset()
                frameInRepeat = 0
                if policy:
                    policy.reset()
                newBest = False
                gameOverUntil = 0
                autoRestart = False

        if now < gameOverUntil:
            draw_world()
            draw_hud()
            draw_game_over()
            pygame.display.flip()
            clock.tick(FPS)
            continue
        if autoRestart:  # network play: start the next game once the game-over panel has been shown
            state = env.reset()
            frameInRepeat = 0
            policy.reset()
            newBest = False
            autoRestart = False

        for _ in range(speed if policy else 1):    # --speed: several game frames per drawn picture (network play only)
            if policy:
                if frameInRepeat == 0:
                    steer, shootOut = policy(state)
                    held = (steer, 1.0 if shootOut > 0.5 else 0.0)
                steer, shoot = held
                frameInRepeat = (frameInRepeat + 1) % repeat
            else:
                keys = pygame.key.get_pressed()
                left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                space = keys[pygame.K_SPACE]
                shoot = 1.0 if (space and not spaceWas) else 0.0
                spaceWas = space
                if left and right:
                    steer = 0.5
                elif left:
                    steer = 0.0
                elif right:
                    steer = 1.0
                else:
                    steer = 0.5
            state, f, done = env.step([steer, shoot])
            if shoot and env.cooldown == BULLET_COOLDOWN_FRAMES - 1:  # only when a bullet actually fired this frame (cooldown was just reset)
                ping("shoot")
            # A bounce is a single-frame velocity-sign flip (gravity-pulled fall
            # becomes an upward launch), not "feet above where they were last
            # frame" -- that stays true for the whole ~20-frame rise of every
            # jump arc and used to fire the bounce sfx on every one of those
            # frames (a machine-gun of plops instead of one bounce per landing).
            bounced = env.velY < 0 <= prevVelY
            # env.stompCount/bulletKillCount/brokenCount are explicit per-step
            # event counts, not list-length diffs -- platforms and monsters also
            # leave their lists when they scroll off the bottom of the screen,
            # which used to fire "crack"/"hit" on that routine cleanup too (the
            # phantom "explosion" a beat after every bounce).
            if env.stompCount > 0:
                ping("stomp")           # stomping a monster (bounce + squish)
            elif bounced:
                ping("bounce")          # platform bounce
            if env.bulletKillCount > 0:
                ping("hit")             # bullet kill
            if env.brokenCount > 0:
                ping("crack")
            prevVelY = env.velY
            if done:
                if env.caughtByMonster:
                    sfx("caught")      # a monster got you (distinct from falling off)
                sfx("die")
                score = round(f)
                newBest = score > best and score > 0
                best = max(best, score)
                gameOverUntil = now + 1200
                autoRestart = policy is not None
                break
        for name in pendingPings:
            sfx(name)
        pendingPings.clear()
        draw_world()
        draw_hud()
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play SprudelJump yourself, or watch a trained network play.")
    parser.add_argument("--model", help="path to a .pt saved by tensorNetwork's saveNetwork(); the network plays instead of the keyboard")
    parser.add_argument("--smoothing", type=float, default=0.96875, help="steer smoothing the model was trained with (default 0.96875 = action repeat 5, 1 = none)")
    parser.add_argument("--repeat", type=int, default=5, help="action repeat the model was trained with: the network decides every n-th frame and its action is held in between (smoothing is applied once per decision)")
    parser.add_argument("--speed", type=int, default=1, help="network play: game frames per drawn picture (e.g. 4 = four times as fast; also the SPEED button at the bottom)")
    a = parser.parse_args()
    main(a.model, a.smoothing, a.repeat, max(1, a.speed))
