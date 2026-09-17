"""Sound effects for the SprudelJump demo.

SFX are genuine 8-bit WAV files from the CC0 (public domain) "8-bit
Platformer SFX" pack by MoxieCat, FamiTracker export:

    https://opengameart.org/content/8-bit-platformer-sfx-0

File mapping (downloaded into sounds/ in this repo, trimmed to punchy
length with a short fade-out):
    bounce  <- spring.wav           platform bounce
    shoot   <- throw.wav            firing a bullet
    stomp   <- warlockhurt.wav      stomping a monster
    hit     <- warlockexploding.wav bullet killing a monster
    crack   <- trapdoor.wav         breakable platform breaking
    die     <- playerhurt.wav       game over

Played through pygame's mixer.  When the mixer is unavailable (headless,
no audio device) load() simply returns {} and play is a silent no-op.
No asset files are fetched at runtime; everything lives in sounds/.
"""
import os

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
_FILE_NAMES = ("bounce", "shoot", "stomp", "hit", "crack", "die")


def load():
    """Return {name: pygame.mixer.Sound} for every available .wav.

    Returns {} when pygame or its mixer is missing — in that case the
    demo stays silent instead of crashing.
    """
    try:
        import pygame
    except ImportError:
        return {}
    try:
        pygame.mixer.init()
    except pygame.error:
        return {}
    bank = {}
    for name in _FILE_NAMES:
        path = os.path.join(_DIR, name + ".wav")
        if not os.path.exists(path):
            continue
        try:
            bank[name] = pygame.mixer.Sound(path)
        except pygame.error:
            pass
    return bank
