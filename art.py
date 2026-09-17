"""Pixel-art sprites for the SprudelJump demo.

Each sprite is a grid of single characters; one character maps to one
solid-color cell ('.' is transparent).  Grids are tiny on purpose -- the
demo scales them up a few times with no smoothing, so the art stays
chunky Doodle-Jump-style while remaining readable at 400x700.
"""

# 12x12 doodle-jumper-style hero, 3x scale -> 36x36 inside the 40x40 hitbox
PLAYER_GRID = [
    "..TTTTTTTT..",
    "..GGGGGGGG..",
    ".GGGGGGGGGG.",
    ".GWWWGGWWWG.",
    ".GWPWGGWPWG.",
    ".GWWWGGWWWG.",
    ".GGGGMMGGGG.",
    ".GGGGGGGGGG.",
    ".GDGGGGGGDG.",
    ".GGGGGGGGGG.",
    ".FFF....FFF.",
    "............",
]
PLAYER_COLORS = {
    "T": (255, 226, 120),  # top highlight
    "G": (238, 200, 60),   # body
    "D": (198, 158, 44),   # side shade
    "W": (255, 255, 255),  # eye white
    "P": (40, 32, 30),     # pupil
    "M": (150, 84, 30),    # mouth
    "F": (245, 135, 50),   # feet
}

# 9x9 grump slime (monster type A), 4x scale -> 36x36 = hitbox
GRUMP_GRID = [
    "..GGGGG..",
    ".GGGGGGG.",
    ".GWWGWWG.",
    ".GPPGPPG.",
    ".GMTMTMG.",
    ".GGGGGGG.",
    ".GGGGGGG.",
    ".KK.GGKK.",
    ".........",
]
GRUMP_COLORS = {
    "G": (105, 195, 85),   # slime body
    "K": (55, 130, 50),    # feet
    "W": (255, 255, 255),  # eye white
    "P": (25, 40, 25),     # pupil
    "M": (60, 25, 45),     # mouth
    "T": (255, 255, 255),  # fangs
}

# 9x9 spiky blob (monster type B), 4x scale -> 36x36 = hitbox
SPIKY_GRID = [
    ".S..S..S.",
    ".SSSSSSS.",
    "SPPPPPPPS",
    "PYYPPYYPP",
    "PPPPPPPPP",
    ".PMMMMMP.",
    "PPPPPPPPP",
    ".PPPPPPP.",
    ".S..S..S.",
]
SPIKY_COLORS = {
    "P": (155, 95, 205),   # body
    "S": (110, 55, 160),   # spikes
    "Y": (255, 220, 70),   # eyes
    "M": (40, 20, 50),     # mouth
}


def _validate(grid):
    w = len(grid[0])
    assert all(len(r) == w for r in grid), \
        f"inconsistent sprite row width in {grid!r}"
    return w


def make_surface(grid, colors, scale):
    """Render a grid to an SRCALPHA pygame surface at the given cell scale."""
    import pygame
    w = _validate(grid)
    surf = pygame.Surface((w * scale, len(grid) * scale), pygame.SRCALPHA)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            color = colors.get(ch)
            if color is not None:
                surf.fill(color + (255,), (x * scale, y * scale, scale, scale))
    return surf


def player_sprite():
    return make_surface(PLAYER_GRID, PLAYER_COLORS, 3)


def monster_sprites():
    return [make_surface(GRUMP_GRID, GRUMP_COLORS, 4),
            make_surface(SPIKY_GRID, SPIKY_COLORS, 4)]
