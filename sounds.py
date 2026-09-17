"""Procedural chiptune-style sound effects for the SprudelJump demo.

All sounds are synthesized at load time from plain math (no asset files,
no numpy, no external dependencies).  They are played through pygame's
mixer; if the mixer is unavailable (headless, no audio device) every
look-up simply returns nothing and the demo stays silent.
"""
import array
import math
import random

SAMPLE_RATE = 24000


def _bounce_samples():
    """Short rising sine blip (280 -> 640 Hz), 80 ms."""
    n = int(SAMPLE_RATE * 0.08)
    phase, out = 0.0, []
    for i in range(n):
        p = i / n
        phase += 2.0 * math.pi * (280.0 + 360.0 * p) / SAMPLE_RATE
        out.append(math.sin(phase) * (1.0 - p))
    return out


def _shoot_samples():
    """Quick descending square-wave zap, 70 ms."""
    n = int(SAMPLE_RATE * 0.07)
    phase, out = 0.0, []
    for i in range(n):
        p = i / n
        phase += 2.0 * math.pi * (950.0 - 450.0 * p) / SAMPLE_RATE
        wave = 1.0 if math.sin(phase) > 0 else -1.0
        out.append(wave * 0.5 * (1.0 - p))
    return out


def _splat_samples():
    """Low thud + noise burst: stomping a monster, 140 ms."""
    n = int(SAMPLE_RATE * 0.14)
    phase, out, rng = 0.0, [], random.Random(7)
    for i in range(n):
        p = i / n
        phase += 2.0 * math.pi * (150.0 - 95.0 * p) / SAMPLE_RATE
        out.append((math.sin(phase) * 0.8 + rng.uniform(-0.6, 0.6) * 0.5)
                   * (1.0 - p) ** 2)
    return out


def _hit_samples():
    """Mid square-wave splat + noise: bullet killing a monster, 100 ms."""
    n = int(SAMPLE_RATE * 0.10)
    phase, out, rng = 0.0, [], random.Random(13)
    for i in range(n):
        p = i / n
        phase += 2.0 * math.pi * (520.0 - 340.0 * p) / SAMPLE_RATE
        wave = 1.0 if math.sin(phase) > 0 else -1.0
        out.append(wave * 0.5 * (1.0 - p) + rng.uniform(-0.4, 0.4) * (1.0 - p) ** 2)
    return out


def _crack_samples():
    """Short noise snap + low bump: breakable platform breaking, 90 ms."""
    n = int(SAMPLE_RATE * 0.09)
    phase, out, rng = 0.0, [], random.Random(29)
    for i in range(n):
        p = i / n
        phase += 2.0 * math.pi * (120.0 - 60.0 * p) / SAMPLE_RATE
        out.append((math.sin(phase) * 0.45 + rng.uniform(-0.7, 0.7) * 0.6)
                   * (1.0 - p) ** 1.5)
    return out


def _die_samples():
    """Falling sawtooth + rumble: game over, 500 ms."""
    n = int(SAMPLE_RATE * 0.5)
    phase, out, rng = 0.0, [], random.Random(11)
    for i in range(n):
        p = i / n
        phase += 2.0 * math.pi * (320.0 * (1.0 - p) ** 1.5 + 40.0) / SAMPLE_RATE
        saw = 2.0 * ((phase / (2.0 * math.pi)) % 1.0) - 1.0
        out.append(saw * 0.6 * (1.0 - p) + rng.uniform(-0.3, 0.3) * (1.0 - p) ** 2)
    return out


_GENERATORS = {
    "bounce": _bounce_samples,
    "shoot": _shoot_samples,
    "splat": _splat_samples,
    "hit": _hit_samples,
    "crack": _crack_samples,
    "die": _die_samples,
}


def _to_sound(samples):
    import pygame
    raw = array.array("h", (int(max(-1.0, min(1.0, s)) * 26000) for s in samples))
    return pygame.mixer.Sound(buffer=raw.tobytes())


def load():
    """Return {name: Sound} or {} when the mixer is unavailable."""
    try:
        import pygame
    except ImportError:
        return {}
    try:
        pygame.mixer.init(SAMPLE_RATE, -16, 1)
    except pygame.error:
        return {}
    bank = {}
    for name, gen in _GENERATORS.items():
        try:
            bank[name] = _to_sound(gen())
        except (pygame.error, ValueError):
            pass
    return bank
