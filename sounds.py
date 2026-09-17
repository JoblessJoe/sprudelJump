"""Sound effects for the SprudelJump demo.

SFX are authentic WAV files ripped from Doodle Jump Arcade, collected by
LeDerpSillyGoober and hosted on The Sounds Resource (fan rip of a 2009
commercial game — kept as demo-only assets; do not ship commercially):

    https://sounds.spriters-resource.com/arcade/doodlejumparcade/asset/450387/

They are converted to 16-bit mono 44100 Hz, trimmed to a punchy length
with a 15 ms fade-out, and normalized to the same peak level so no
single sound drowns out the others.  File mapping:
    bounce  <- DJ_Jump.wav           platform bounce (short "plop")
    shoot   <- rocket.wav            firing a bullet (launch burst, 0.35 s)
    stomp   <- jumponmonster.wav     stomping a monster
    hit     <- monster-crash.wav     bullet killing a monster
    crack   <- explodingplatform.wav breakable platform breaking
    die     <- Start_Failure.wav     game over

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
