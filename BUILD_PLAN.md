# SprudelJump Environment — Build Plan

(SprudelJump = this project's Doodle Jump clone. Name the main class `SprudelJumpEnv`.)

## Scope (read this first)

Build ONLY a game **environment** — not an AI, not a training loop, not UI polish.
This environment will later be controlled by an external neural network that calls
`step(action)` in a loop, many thousands of times, headless (no rendering). Keep it
simple, keep it lean, follow the milestones below in order.

**Explicitly out of scope — do not build these:**
- Enemies, power-ups, moving platforms, menus, sound
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
class SprudelJumpEnv:
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
            per-step delta — see "Scoring" in step logic below). This IS the point
            counter/score — there is no separate score value.
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
   `[0.0, 1.0]` (meaning "no platform, far away"). Platform *width* is deliberately
   not part of the state (see Difficulty scaling below) — the network only ever
   learns from position, not width, same as a human wouldn't get a numeric readout.

## Physics constants — use these exact values, don't invent your own

```python
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
PLATFORM_WIDTH = 60        # width of a platform spawned at difficulty 0 (see scaling below)
PLATFORM_HEIGHT = 15
GRAVITY = 0.4               # px/frame^2, downward
BOUNCE_VELOCITY = -13.0     # px/frame, upward (negative y = up, screen coords)
MAX_FALL_SPEED = 15.0
HORIZONTAL_SPEED = 6.0      # px/frame, max horizontal speed
SCROLL_THRESHOLD_Y = SCREEN_HEIGHT * 0.4
PLATFORM_GAP_MIN = 80       # vertical gap between platforms at difficulty 0
PLATFORM_GAP_MAX = 130
```

## Platform representation

Each platform is `(x, y, width)` — **not** just `(x, y)`. Store its own `width` at
spawn time (see Difficulty scaling — older, lower platforms keep the easier width
they were spawned with; only new platforms get narrower as height increases). Use
each platform's own `width` field, not the `PLATFORM_WIDTH` constant, everywhere a
platform's width matters (collision x-overlap check, state vector calc if ever
needed, rendering). The `PLATFORM_WIDTH` constant is only the *starting* value used
at difficulty 0.

## Difficulty scaling — gets harder the higher you climb, like the real game

```python
DIFFICULTY_MAX_HEIGHT = 20000   # self.totalHeight at which difficulty caps at 1.0
PLATFORM_GAP_MIN_HARD = 100     # gap range at max difficulty (vs 80 at difficulty 0)
PLATFORM_GAP_MAX_HARD = 160     # (vs 130 at difficulty 0)
PLATFORM_WIDTH_MIN = 40         # narrowest platform width at max difficulty (vs 60)
```

Every time a new platform is spawned (in the housekeeping step below), compute:
```python
difficultyT = min(1.0, self.totalHeight / DIFFICULTY_MAX_HEIGHT)
gapMin = PLATFORM_GAP_MIN + difficultyT * (PLATFORM_GAP_MIN_HARD - PLATFORM_GAP_MIN)
gapMax = PLATFORM_GAP_MAX + difficultyT * (PLATFORM_GAP_MAX_HARD - PLATFORM_GAP_MAX)
platformWidth = PLATFORM_WIDTH - difficultyT * (PLATFORM_WIDTH - PLATFORM_WIDTH_MIN)
newY = topmostPlatform.y - random.uniform(gapMin, gapMax)
newPlatform = (x=random.uniform(0, SCREEN_WIDTH - platformWidth), y=newY, width=platformWidth)
```
This only affects *newly spawned* platforms — it's a smooth, capped, linear ramp
based on `self.totalHeight`, nothing more elaborate. No moving/breakable platforms,
no separate difficulty state to track beyond `self.totalHeight`, which you already
have.

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
   top this frame AND the player's x-range overlaps the platform's own x-range
   (using its stored `width`), then `player.velY = BOUNCE_VELOCITY`.
5. **Scrolling**: if `player.y < SCROLL_THRESHOLD_Y`: `scrollAmount =
   SCROLL_THRESHOLD_Y - player.y`; add `scrollAmount` to `self.totalHeight` (this IS
   the score/fitness/point counter); set `player.y = SCROLL_THRESHOLD_Y`; shift
   every platform's `y += scrollAmount`.
6. **Platform housekeeping**: remove any platform with `platform.y > SCREEN_HEIGHT`.
   While the topmost remaining platform's `y > 0`, spawn a new one above it using
   the Difficulty scaling formula above.
7. **Game over**: if `player.y > SCREEN_HEIGHT`, `done = True`.
8. Return `(self._getState(), self.totalHeight, done)`.

## `reset()` logic

- Player starts at `x = SCREEN_WIDTH / 2`, `y = SCREEN_HEIGHT / 2`, `velY = 0`.
- `self.totalHeight = 0.0`.
- Generate ~8 initial platforms (all at `width = PLATFORM_WIDTH`, difficulty 0, since
  `totalHeight` starts at 0) filling the screen from `y = SCREEN_HEIGHT` up to
  `y = 0`, spaced by `random.uniform(PLATFORM_GAP_MIN, PLATFORM_GAP_MAX)`, random x
  each.
- Force one platform directly under the player's start position so it doesn't
  immediately fall through nothing: `y = player.y + PLAYER_HEIGHT + 10`,
  `x = player.x - PLATFORM_WIDTH / 2`, `width = PLATFORM_WIDTH`.
- Return `self._getState()`.

## Build order — follow exactly, in order. Run each self-check before moving on.

**M1 — Gravity + one fixed platform. No scrolling, no wrap, no horizontal input yet.**
Self-check: call `step()` 200 times with `action = 0.5`, print `player.y` every 20
steps. Should oscillate (fall, bounce, fall, bounce), never diverge.

**M2 — Add horizontal movement + screen wrap.**
Self-check: run 100 steps with `action = 1.0`, confirm `player.x` increases then
wraps to a negative value once past `SCREEN_WIDTH`.

**M3 — Multiple platforms + real collision (only bounces when actually landing on one).**
Self-check: place 3 platforms at different x/y (each with a `width`), run 300 steps
with random actions, count and print how many bounces occurred (should be several,
not zero, not every single frame).

**M4 — Scrolling + platform spawn/removal + difficulty scaling.**
Self-check: run 3000 steps with `action = 0.5`, print `self.totalHeight` every 300
steps (should be non-decreasing, and actually grow over the run), AND print the
average gap/width of newly spawned platforms in the first 500 steps vs. the last 500
steps — the later average gap should be visibly larger / width visibly smaller than
the earlier one, confirming difficulty scaling is actually doing something.

**M5 — Full `reset()`/`step()`/`done`, matching the interface contract exactly.**
Self-check:
```python
env = SprudelJumpEnv()
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
Minimal `pygame`: draw player + platforms as rectangles (vary rectangle width by
each platform's own `width`), call `env.step()` each frame (random action, or
keyboard-driven for manual play), redraw, and draw `env.totalHeight` as on-screen
text (this is the score). Not used for training. Keep it under ~100 lines. Skip
entirely if short on context — `env.py` is what actually matters.

## Working instructions — read carefully, this will run unattended for hours

- **Work completely independently. Do not stop to ask questions, do not wait for
  confirmation between milestones, do not pause for approval.** This is expected to
  take several hours of continuous, unattended work — budget your time and context
  accordingly across all milestones rather than spending it all on the first one.
- Follow milestones **in order** — don't build M4 before M1-M3 are confirmed working.
- After a milestone's self-check passes, move on immediately. Don't add polish or
  features not listed here.
- If a self-check's output looks wrong: debug and fix it yourself, then re-run the
  self-check, before moving to the next milestone. Don't carry a suspected bug
  forward. Don't stop and wait to be told what to do.
- If you get genuinely stuck on the same problem after several real fix attempts:
  write a short, specific note into `STATUS.md` describing exactly what's wrong and
  what you tried, then continue on to whatever you still can do (e.g. skip to
  writing the self-check script for a later milestone against your best-effort code)
  rather than stopping entirely.
- Minimal comments. One-liners only where the WHY is non-obvious — no docstrings, no
  restating what the code already says.
- No `pygame` import anywhere except `demo_render.py`. `env.py` must run with just
  the standard library, headless, no display needed.
- Don't stop to re-derive Doodle Jump mechanics from scratch or second-guess the
  numbers given here — this document is the complete spec. For anything not
  explicitly covered, make the simplest reasonable choice and keep moving.
- When every milestone you attempted is done, write a final short summary into
  `STATUS.md`: what was built, what each self-check actually printed, and anything
  you weren't fully sure about.
