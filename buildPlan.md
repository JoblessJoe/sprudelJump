# Doodle Jump Environment — Build Plan

## Scope (read this first)

Build ONLY a game **environment** — not an AI, not a training loop, not UI polish.
This environment will later be controlled by an external neural network that calls
`step(action)` in a loop, many thousands of times, headless (no rendering). Keep it
simple, keep it lean, follow the milestones below in order.

**Explicitly out of scope — do not build these:**
- Enemies, power-ups, moving platforms, score UI, menus, sound
- The neural network / training code (a separate project handles that)
- Multiprocessing / parallel execution (also handled separately)

**Deliverables (two files):**
- `env.py` — the game environment. Pure Python + `random` only. No external
  dependencies. Must run with zero setup.
- `demo_render.py` — OPTIONAL, build this LAST, only if time/context remains. A
  simple `pygame` script to visually watch the env play out, for human
  sanity-checking. Never used for training — skip it entirely if you're running low
  on context, `env.py` is the only file that actually matters.

## Interface contract — do not deviate from this shape

```python
class DoodleJumpEnv:
    def reset(self) -> list[float]:
        """Resets the game to a fresh start. Returns the initial state vector (see below)."""

    def step(self, action: float) -> tuple[list[float], float, bool]:
        """
        action: float in [0.0, 1.0] — this comes directly from a sigmoid-output
        neural network, so it MUST be exactly this range.
        0.0 = full left, 0.5 = no horizontal movement, 1.0 = full right.

        Returns (state, fitness, done):
          - state: new state vector, same format as reset()
          - fitness: the CURRENT total height climbed so far (running total, not a
            per-step delta — see "Scoring" in step logic below)
          - done: True if the player fell off the bottom of the screen (game over)
        """

    def render(self):
        """OPTIONAL. Only used by demo_render.py. Leave as `pass` in env.py itself."""
```

## State vector — exact composition, exact order, exactly 12 floats

1. `playerX / SCREEN_WIDTH` (normalized, roughly [0,1])
2. `playerVelY / MAX_FALL_SPEED`, clamped to [-1, 1]
3-12. For the **5 nearest platforms above the player** (sorted by vertical distance,
closest first), 2 floats each, in this order:
   - relative X: `(platform.x - player.x) / SCREEN_WIDTH`, clamped [-1, 1]
   - relative Y: `(player.y - platform.y) / SCREEN_HEIGHT`, clamped [0, 1]

   If fewer than 5 platforms exist above the player, pad remaining slots with
   `[0.0, 1.0]` (meaning "no platform, far away").

## Physics constants — use these exact values, don't invent your own

```python
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
PLATFORM_WIDTH = 60
PLATFORM_HEIGHT = 15
GRAVITY = 0.4              # px/frame^2, downward
BOUNCE_VELOCITY = -13.0    # px/frame, upward (negative y = up, screen coords)
MAX_FALL_SPEED = 15.0
HORIZONTAL_SPEED = 6.0     # px/frame, max horizontal speed
SCROLL_THRESHOLD_Y = SCREEN_HEIGHT * 0.4
PLATFORM_GAP_MIN = 80
PLATFORM_GAP_MAX = 130
```

## Action → horizontal velocity

```python
horizontalVelocity = (action - 0.5) * 2 * HORIZONTAL_SPEED
```

## Screen wrap (horizontal)

If `player.x < -PLAYER_WIDTH`: `player.x = SCREEN_WIDTH`.
If `player.x > SCREEN_WIDTH`: `player.x = -PLAYER_WIDTH`.

## Core `step()` logic, in this exact order

1. Apply `horizontalVelocity` to `player.x`, then apply screen wrap.
2. Gravity: `player.velY += GRAVITY`, clamp to `MAX_FALL_SPEED`.
3. `player.y += player.velY`.
4. **Collision** — only check when `player.velY > 0` (falling, not rising): for each
   platform, if the player's feet (`player.y + PLAYER_HEIGHT`) cross the platform's
   top this frame AND the player's x-range overlaps the platform's x-range, then
   `player.velY = BOUNCE_VELOCITY`.
5. **Scrolling**: if `player.y < SCROLL_THRESHOLD_Y`: `scrollAmount =
   SCROLL_THRESHOLD_Y - player.y`; add `scrollAmount` to `self.totalHeight` (this IS
   the score/fitness); set `player.y = SCROLL_THRESHOLD_Y`; shift every platform's
   `y += scrollAmount`.
6. **Platform housekeeping**: remove any platform with `platform.y > SCREEN_HEIGHT`.
   While the topmost remaining platform's `y > 0`, spawn a new one above it:
   `newY = topmostPlatform.y - random.uniform(PLATFORM_GAP_MIN, PLATFORM_GAP_MAX)`,
   `newX = random.uniform(0, SCREEN_WIDTH - PLATFORM_WIDTH)`.
7. **Game over**: if `player.y > SCREEN_HEIGHT`, `done = True`.
8. Return `(self._getState(), self.totalHeight, done)`.

## `reset()` logic

- Player starts at `x = SCREEN_WIDTH / 2`, `y = SCREEN_HEIGHT / 2`, `velY = 0`.
- `self.totalHeight = 0.0`.
- Generate ~8 initial platforms filling the screen from `y = SCREEN_HEIGHT` up to
  `y = 0`, spaced by `random.uniform(PLATFORM_GAP_MIN, PLATFORM_GAP_MAX)`, random x
  each.
- Force one platform directly under the player's start position so it doesn't
  immediately fall through nothing: `y = player.y + PLAYER_HEIGHT + 10`,
  `x = player.x - PLATFORM_WIDTH / 2`.
- Return `self._getState()`.

## Build order — follow exactly, in order. Run each self-check before moving on.

**M1 — Gravity + one fixed platform. No scrolling, no wrap, no horizontal input yet.**
Self-check: call `step()` 200 times with `action = 0.5`, print `player.y` every 20
steps. Should oscillate (fall, bounce, fall, bounce), never diverge.

**M2 — Add horizontal movement + screen wrap.**
Self-check: run 100 steps with `action = 1.0`, confirm `player.x` increases then
wraps to a negative value once past `SCREEN_WIDTH`.

**M3 — Multiple platforms + real collision (only bounces when actually landing on one).**
Self-check: place 3 platforms at different x/y, run 300 steps with random actions,
count and print how many bounces occurred (should be several, not zero, not every
single frame).

**M4 — Scrolling + platform spawn/removal.**
Self-check: run 1000 steps with `action = 0.5`, print `self.totalHeight` every 100
steps. Should be non-decreasing, and actually grow over the run (not stay at 0).

**M5 — Full `reset()`/`step()`/`done`, matching the interface contract exactly.**
Self-check:
```python
env = DoodleJumpEnv()
s = env.reset()
assert len(s) == 12
steps = 0
done = False
while not done and steps < 5000:
    s, fitness, done = env.step(random.random())
    steps += 1
print(steps, fitness)
```
Confirm it terminates (game over is reachable) and `fitness` ends up a sane,
positive-ish number.

**M6 (optional, last — only if time/context remains) — `demo_render.py`.**
Minimal `pygame`: draw player + platforms as rectangles, call `env.step()` each
frame (random action, or keyboard-driven for manual play), redraw. Not used for
training. Keep it under ~80 lines. Skip entirely if short on context — `env.py` is
what actually matters.

## Working instructions

- Follow milestones **in order** — don't build M4 before M1-M3 are confirmed working.
- After a milestone's self-check passes, move on immediately. Don't add polish or
  features not listed here.
- Minimal comments. One-liners only where the WHY is non-obvious — no docstrings, no
  restating what the code already says.
- No `pygame` import anywhere except `demo_render.py`. `env.py` must run with just
  the standard library, headless, no display needed.
- If a self-check's output looks wrong, fix that milestone before moving on — don't
  carry a suspected bug forward into the next one.
- Don't stop to ask clarifying questions or re-derive Doodle Jump mechanics from
  scratch — this document is the complete spec. For anything not explicitly covered,
  make the simplest reasonable choice and keep moving.
