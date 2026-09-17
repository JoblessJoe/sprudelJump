# SprudelJump Environment — Core Spec

(SprudelJump = this project's Doodle Jump clone. Name the main class `SprudelJumpEnv`.)

This is the shared spec every milestone builds on. Read it once per fresh
context, then go straight to your current milestone file in `plan/M*.md` —
do not re-read completed milestone files, and do not re-read this file again
in the same context unless it was compacted away.

## Scope (read this first)

Build ONLY a game **environment** — not an AI, not a training loop, not UI polish.
This environment will later be controlled by an external neural network that calls
`step(action)` in a loop, many thousands of times, headless (no rendering). Keep it
simple, keep it lean.

**Explicitly out of scope — do not build these:**
- Power-ups, moving platforms, menus, sound
- The neural network / training code (a separate project handles that)
- Multiprocessing / parallel execution (also handled separately)
- Aiming — shooting is always straight up, no direction control

(Monsters, shooting, and breakable platforms ARE in scope as of `plan/M7.md`
through `plan/M10.md` — see the updated interface, state vector, and constants
below, which describe the FINAL target shape once all of M7-M10 are done, same
as this file always described the target for M1-M6. Built incrementally:
M7 adds breakable platforms (state → 17 floats), M8 adds monsters (→ 23
floats), M9 changes `action` from one float to `[steer, shoot]` and adds
bullets, M10 updates `demo_render.py`. Go to whichever `plan/M*.md` is next
after your last completed one.)

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

    def step(self, action: list[float]) -> tuple[list[float], float, bool]:
        """
        action: [steer, shoot], a 2-element list, both floats in [0.0, 1.0] — this
        comes directly from a 2-output sigmoid neural network, so both MUST be
        exactly this range.
          - action[0] (steer): 0.0 = full left, 0.5 = no horizontal movement, 1.0 = full right
          - action[1] (shoot): > 0.5 fires a bullet this frame (subject to cooldown,
            see Monsters & shooting below). No aiming — always straight up.

        Returns (state, fitness, done):
          - state: new state vector, same format as reset()
          - fitness: the CURRENT total height climbed so far, PLUS any monster-kill
            bonuses earned so far (running total, not a per-step delta — see
            "Scoring" in step logic below). This IS the point counter/score — there
            is no separate score value.
          - done: True if the player fell off the bottom of the screen, OR was hit
            by a monster (game over either way)
        """

    def render(self):
        """OPTIONAL. Only used by demo_render.py. Leave as `pass` in env.py itself."""
```

## State vector — exact composition, exact order, exactly 23 floats

1. `playerX / SCREEN_WIDTH` (normalized, roughly [0,1])
2. `playerVelY / MAX_FALL_SPEED`, clamped to [-1, 1]
3-17. For the **5 nearest platforms above the player** (sorted by vertical distance,
closest first), 3 floats each, in this order:
   - relative X: `(platform.x - player.x) / SCREEN_WIDTH`, clamped [-1, 1]
   - relative Y: `(player.y - platform.y) / SCREEN_HEIGHT`, clamped [0, 1]
   - breakable flag: `1.0` if this platform is breakable, else `0.0`

   If fewer than 5 platforms exist above the player, pad remaining slots with
   `[0.0, 1.0, 0.0]` (meaning "no platform, far away, not breakable"). Platform
   *width* is deliberately not part of the state (see Difficulty scaling below) —
   the network only ever learns from position (and now breakability), not width,
   same as a human wouldn't get a numeric readout.
18-23. For the **3 nearest monsters above the player** (sorted by vertical
distance, closest first), 2 floats each, same treatment as platforms:
   - relative X: `(monster.x - player.x) / SCREEN_WIDTH`, clamped [-1, 1]
   - relative Y: `(player.y - monster.y) / SCREEN_HEIGHT`, clamped [0, 1]

   Pad with `[0.0, 1.0]` per missing monster, same as platforms. Bullets are
   deliberately NOT part of the state — they're a short-lived side effect of the
   shoot action, not something the network needs to track positionally.

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
MONSTER_WIDTH = 36
MONSTER_HEIGHT = 36
MONSTER_SPAWN_CHANCE_BASE = 0.0    # chance a newly spawned platform also gets a monster, at difficulty 0
MONSTER_SPAWN_CHANCE_MAX = 0.35    # same, at difficulty 1.0 (linear ramp, same difficultyT as platform gap)
BULLET_WIDTH = 6
BULLET_HEIGHT = 14
BULLET_SPEED = 10.0          # px/frame, upward
BULLET_COOLDOWN_FRAMES = 10  # min frames between shots
MONSTER_KILL_BONUS = 100.0   # added directly to self.totalHeight on any kill (shot or stomped)
```

## Platform representation

Each platform is `(x, y, width, breakable)` — **not** just `(x, y)`. Store its own
`width` at spawn time (see Difficulty scaling — older, lower platforms keep the
easier width they were spawned with; only new platforms get narrower as height
increases). Use each platform's own `width` field, not the `PLATFORM_WIDTH`
constant, everywhere a platform's width matters (collision x-overlap check, state
vector calc if ever needed, rendering). The `PLATFORM_WIDTH` constant is only the
*starting* value used at difficulty 0. `breakable` is a plain `bool`, decided once
at spawn time (see Breakable platforms below).

## Breakable platforms

```python
BREAKABLE_CHANCE_BASE = 0.05   # chance a newly spawned platform is breakable, at difficulty 0
BREAKABLE_CHANCE_MAX = 0.30    # same, at difficulty 1.0 (same difficultyT ramp as everything else)
```
When spawning a new platform (housekeeping step below), after computing
`difficultyT`: `breakable = random.random() < (BREAKABLE_CHANCE_BASE + difficultyT * (BREAKABLE_CHANCE_MAX - BREAKABLE_CHANCE_BASE))`.

Behavior: a breakable platform bounces the player exactly like a normal one
(`BOUNCE_VELOCITY`, no different physics) — but immediately after that bounce is
applied, remove it from `self.platforms`. One use, then gone. The initial ~8
platforms generated in `reset()` are all `breakable=False` (keep the opening of
every episode predictable and safe).

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
breakable = random.random() < (BREAKABLE_CHANCE_BASE + difficultyT * (BREAKABLE_CHANCE_MAX - BREAKABLE_CHANCE_BASE))
newPlatform = (x=random.uniform(0, SCREEN_WIDTH - platformWidth), y=newY, width=platformWidth, breakable=breakable)
```
This only affects *newly spawned* platforms — it's a smooth, capped, linear ramp
based on `self.totalHeight`, nothing more elaborate. No moving platforms, no
separate difficulty state to track beyond `self.totalHeight`, which you already
have (breakability is its own independent roll, see Breakable platforms above).

## Monster representation

Each monster is `(x, y)` — `MONSTER_WIDTH`/`MONSTER_HEIGHT` are constant, no
per-monster width like platforms have. A monster is either alive (in
`self.monsters`) or dead (removed) — no other state.

## Monster spawning (extends platform housekeeping, step 6 below)

Whenever a new platform is spawned, also roll for a monster on that same platform:
```python
difficultyT = min(1.0, self.totalHeight / DIFFICULTY_MAX_HEIGHT)  # same t as platform gap/width
spawnChance = MONSTER_SPAWN_CHANCE_BASE + difficultyT * (MONSTER_SPAWN_CHANCE_MAX - MONSTER_SPAWN_CHANCE_BASE)
if random.random() < spawnChance:
    monsterX = newPlatform.x + newPlatform.width / 2 - MONSTER_WIDTH / 2
    monsterY = newPlatform.y - MONSTER_HEIGHT
    self.monsters.append([monsterX, monsterY])
```

## Bullets and firing

`self.bullets` is a list of `[x, y]` pairs. Track `self.cooldown` (frames until
next shot allowed), decrementing by 1 each `step()` call, floored at 0.

Each `step()`, if `action[1] > 0.5` and `self.cooldown == 0`:
```python
self.bullets.append([player.x + PLAYER_WIDTH / 2 - BULLET_WIDTH / 2, player.y])
self.cooldown = BULLET_COOLDOWN_FRAMES
```
Every `step()`, regardless of firing: move each bullet up (`bullet.y -= BULLET_SPEED`),
remove any bullet with `y < -BULLET_HEIGHT`.

## Action → horizontal velocity

```python
horizontalVelocity = (action[0] - 0.5) * 2 * HORIZONTAL_SPEED
```

## Screen wrap (horizontal)

If `player.x < -PLAYER_WIDTH`: `player.x = SCREEN_WIDTH`.
If `player.x > SCREEN_WIDTH`: `player.x = -PLAYER_WIDTH`.

## Core `step()` logic, in this exact order

1. Apply `horizontalVelocity` to `player.x`, then apply screen wrap.
2. Gravity: `player.velY += GRAVITY`, clamp to `MAX_FALL_SPEED`.
3. `player.y += player.velY`.
4. **Platform collision** — only check when `player.velY > 0` (falling, not
   rising): for each platform, if the player's feet (`player.y + PLAYER_HEIGHT`)
   cross the platform's top this frame AND the player's x-range overlaps the
   platform's own x-range (using its stored `width`), then `player.velY =
   BOUNCE_VELOCITY`; if that platform's `breakable` is `True`, remove it from
   `self.platforms` right after applying the bounce.
5. **Monster stomp/contact** — for each living monster: if `player.velY > 0`
   (falling) AND the feet-cross-top test (same shape as platform collision, using
   `MONSTER_WIDTH`/`MONSTER_HEIGHT`) passes: kill it (remove from `self.monsters`,
   `self.totalHeight += MONSTER_KILL_BONUS`), and `player.velY = BOUNCE_VELOCITY`
   (a stomp also bounces, same as a platform). Otherwise, if the player's full
   bounding box overlaps that monster's bounding box at all (any direction, not
   just falling), `done = True` — this is a fatal hit.
6. **Firing**: if `action[1] > 0.5` and `self.cooldown == 0`, spawn a bullet (see
   Bullets and firing above) and reset `self.cooldown`. Then decrement
   `self.cooldown` by 1 (floor 0) regardless of whether a shot was just fired.
7. **Bullet movement + bullet/monster collision**: move every bullet up by
   `BULLET_SPEED`, remove any bullet with `y < -BULLET_HEIGHT`. For every
   remaining bullet, check overlap against every living monster — on a hit,
   remove both the bullet and the monster, `self.totalHeight += MONSTER_KILL_BONUS`.
8. **Scrolling**: if `player.y < SCROLL_THRESHOLD_Y`: `scrollAmount =
   SCROLL_THRESHOLD_Y - player.y`; add `scrollAmount` to `self.totalHeight` (this IS
   the score/fitness/point counter, on top of any kill bonuses already added this
   step); set `player.y = SCROLL_THRESHOLD_Y`; shift every platform's, every
   monster's, AND every bullet's `y += scrollAmount`.
9. **Platform/monster housekeeping**: remove any platform or monster with
   `y > SCREEN_HEIGHT`. While the topmost remaining platform's `y > 0`, spawn a
   new one above it using the Difficulty scaling + Breakable platforms formulas,
   then roll for a monster on it per Monster spawning above.
10. **Game over**: if `player.y > SCREEN_HEIGHT`, `done = True` (in addition to
    the monster-contact case in step 5 — either one ends the episode).
11. Return `(self._getState(), self.totalHeight, done)`.

## `reset()` logic

- Player starts at `x = SCREEN_WIDTH / 2`, `y = SCREEN_HEIGHT / 2`, `velY = 0`.
- `self.totalHeight = 0.0`, `self.monsters = []`, `self.bullets = []`, `self.cooldown = 0`.
- Generate ~8 initial platforms (all at `width = PLATFORM_WIDTH`, `breakable =
  False`, difficulty 0, since `totalHeight` starts at 0) filling the screen from
  `y = SCREEN_HEIGHT` up to `y = 0`, spaced by `random.uniform(PLATFORM_GAP_MIN,
  PLATFORM_GAP_MAX)`, random x each. No monsters spawned on these initial ones —
  keep the opening safe.
- Force one platform directly under the player's start position so it doesn't
  immediately fall through nothing: `y = player.y + PLAYER_HEIGHT + 10`,
  `x = player.x - PLATFORM_WIDTH / 2`, `width = PLATFORM_WIDTH`, `breakable = False`.
- Return `self._getState()`.

## Working instructions — this runs unattended for hours

- **Work completely independently. Do not stop to ask questions, do not wait for
  confirmation between milestones, do not pause for approval.** Budget your time
  and context across all milestones rather than spending it all on the first one.
- Follow milestones **in order** — don't build M4 before M1-M3 are confirmed working.
- After a milestone's self-check passes, move on immediately to the next `plan/M*.md`
  file. Don't add polish or features not listed there.
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
