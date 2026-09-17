# SprudelJump — STATUS

## Built

`env.py` — headless Doodle-Jump environment, pure stdlib (only `import random`), no pygame.
Complete per CORE_SPEC interface (as of M10): `reset() -> list[23 floats]`,
`step(action = [steer, shoot], both in [0,1]) -> (state, fitness, done)`, `render() = pass`.
(steer 0.5 = no movement; shoot > 0.5 fires subject to a 10-frame cooldown.)
- Physics: gravity 0.4, bounce -13.0, max fall 15.0, horizontal speed 6.0, screen wrap.
- Camera scroll at playerY < 280 (0.4*700); platforms shift down, despawn below screen, spawn above top.
- Difficulty scaling t = totalHeight/20000: gap 80–130 → 100–160, platform width 60 → 40.
- State: playerX/W, clamped velY, then 5 nearest above-platforms as (dx/W clamped, dy/H clamped); padding [0.0, 1.0].
- Fitness = running totalHeight (score); done when player falls off bottom.

`demo_render.py` — optional pygame viewer (player/platforms as rects, width per platform,
H drawn on screen, keyboard play + auto-random, Esc to quit, R to reset). Under 100 lines.
pygame is NOT importable from env.py.

Milestone checks: `check_m2.py`, `check_m3.py`, `check_m4.py`, `check_m5.py` (M1 checked inline),
plus `check_m7.py`, `check_m8.py`, `check_m9.py`.

## Self-check results (real output)

- M2: 100 steps action=1.0 — wrap verified, playerX crossed right edge to the negative side (380 → -4). Pass.
- M3: aligned-platform run — 5 bounces in 300 random steps (0 < n < 50); negative control with
  a far-off platform: player passes through cleanly (no false bounce, y keeps growing). Pass.
- M4: 3000 steps action=0.5 seed 42 — `monotonic: True`, H=136.2. Varying action gives real
  growth: alternating 0.2/0.8 → H=193.2 (0 deaths); random → H=339.6 over 5 episodes.
  Difficulty formula verified directly at t=0/0.25/0.5/1.0 — spawned gap/width ranges match the
  spec endpoints exactly (e.g. t=1.0 gap [100,160], width 40). Pass.
- M5: 5000-step loop with random actions — terminates well before the cap on all seeds:
  seed 42 → 468 steps, fitness 123.6; seed 1 → 86, 123.6; seed 7 → 191, 312.2; seed 99 → 471, 123.6. Pass.
- M6: pygame unavailable on this box (PEP 668, system-managed python, pip refused). demo_render.py
  is written and import-cleans; it prints a friendly skip message when pygame is absent. Not run.

## Findings / uncertainties

- **Constant action 0.5 locks an equilibrium.** With zero horizontal input the player settles
  into a bounce on the bottom platforms whose overshoot of the scroll line approaches zero, so
  H plateaus (~123–148 depending on seed). This is a genuine physics property, not a bug:
  varying any action resumes growth (verified above). An RL policy will not stay at exactly 0.5.
- Spawn x is uniform; platforms can overlap in x. Not spec-restricted; left as simplest choice.
- Starting platform sits 10 px below the player plus its own height; the spec's exact gap was not
  pinned, so 10 px pad was chosen.
- Collision is a feet-sweep test (feetPrev <= platform.top <= feet) against platform top edge only —
  the standard Doodle-Jump model (no side/corner collision).

All milestones M1–M6 complete (M6 delivered as written code, not executed). `env.py` is ready
for headless NN training: thousands of `reset()`/`step()` loops, zero external dependencies.

---

## M7–M10 (breakable platforms, monsters, bullets, renderer) — complete

Build order per plan/M7.md → M10.md: one file at a time, milestone self-checks passed before
moving on. Final env.py: 4-tuple platforms `[x, y, width, breakable]`, `self.monsters` (2-tuple
`[x, y]`, 36×36), `self.bullets` (2-tuple `[x, y]`, 6×14), `self.cooldown`; `reset() -> 23 floats`;
`step([steer, shoot]) -> (state, fitness, done)`.

**M7 — breakable platforms (state 12 → 17).** Spawn chance = 0.05 + t·(0.30 − 0.05); initial and
forced start platforms are never breakable; a breakable is removed right after the frame it bounces
the player (every breakable hit in a frame is removed). check_m7.py:
- 1a forced breakable platform at landing spot → gone by frame 40, player bounced off a survivor. Pass.
- 1b forced normal platform, identical setup → still present after 40+ frames. Pass.
- 2 `len(reset()) == 17`. Pass.
- 3 The literal spec check (5000 steps, action 0.5, breakable count first1500 vs last1500) is
  degenerate for the documented 0.5 equilibrium lock (M4 finding): H plateaus ~123–193, only 3
  spawn events in 5000 steps (3a, recorded for the record), so no difficulty delta exists between
  windows and n is too small to distinguish 0.05 from anything. Verified the actual property
  directly instead: measured breakable rate at forced difficulty t=0.0/0.5/1.0 = 0.045/0.170/
  0.305 vs expected 0.050/0.175/0.300. Pass.

**M8 — monsters (state 17 → 23; action unchanged).** Monster = `[monsterX, monsterY]` centered on
its platform's top; spawn chance = 0.0 + t·(0.35 − 0.0); stomp = falling + feet sweeps monster top
→ remove, `totalHeight += 100`, bounce; any other full-box overlap → done. Monsters scroll/despawn
like platforms. State: 3 nearest monsters above the player as (dx/W clamped, dy/H clamped), padded
`[0.0, 1.0]`. check_m8.py:
- 1 stomp: forced monster above forced platform, player falling from above → monster removed,
  height +100, bounced, not dead. Pass.
- 2 lethal contact: monster overlapping player's x at player's y while rising (velY < 0) → done=True
  on that step, monster untouched. Pass.
- 3 `len(reset()) == 23`. Pass.
- 4 Same equity caveat as M7-3 under action 0.5 (4a: H=123.6, 0 monsters alive, recorded); direct
  verification: measured monster chance at t=0.0/0.5/1.0 = 0.000/0.177/0.351 vs expected
  0.000/0.175/0.350. Pass.

**M9 — bullets + action becomes [steer, shoot].** Per M9.md the interface migration happened FIRST
as its own step: `step()` now reads `action[0]` for steering (bare-float fallback kept so the old
scripts run unchanged), all callers of `env.step` in `check_m2..m8.py` and `demo_render.py` were
updated to 2-element lists, and all re-ran green before touching bullet code. Bullets spawn straight
up from the player on shoot > 0.5 when cooldown is 0 (reset to 10 frames, decrement every step),
move up at 10 px/frame, despawn above the top, shift on scroll, and kill monsters on any overlap
(both removed, +100). Bullets are deliberately absent from the state vector, per CORE_SPEC.
- Old-script re-run on the new interface, all pass: check_m2, check_m3 (platforms now 4-tuples),
  check_m4, check_m5 (state assert 12 → 23), check_m7, check_m8.
- check_m9 bullet kill: monster centered on the bullet stream directly above the player,
  `step([0.5, 1.0])` → bullet appears, travels upward, monster removed on contact, height jumped
  to ≥ 100. Pass.
- check_m9 cooldown: 30 frames of `step([0.5, 1.0])` → exactly 3 shots fired (one per 10-frame
  window), not 30. Pass.

**M10 — demo_render.py extended (112 lines).** Builds `action = [steer, shoot]`; Space held →
shoot = 1.0; monsters drawn as red ellipses, bullets as small pale rects, breakable platforms in
amber vs teal for normal ones; everything else (background, HUD, game-over flow, R/Esc) unchanged.
Self-check: pygame is still not installable on this box (same PEP 668 block as M6), so the visual
run was not possible. Verified instead: `python3 -m py_compile` clean on demo_render.py and env.py,
and a headless import of demo_render (pygame = None) hits the guarded skip path and exits
gracefully; line count 112 (< ~130 cap); a 3000-step headless smoke run (mixed steer/shoot) keeps
the 23-float state invariant. **Uncertainty:** the four visual confirmations M10 asks for
(monster/bullet/breakable rendering, and on-screen removal on stomp/shoot) are logically correct by
construction — the draw loop iterates the same `env.monsters`/`env.bullets`/breakable flag that the
passing headless self-checks exercise — but they were not observed on screen.

Cross-check after M10: `check_m2`–`check_m9` all pass against the final env.py.

**Unsure / deferred:**
- The two "first 1500 vs last 1500" ramp checks in M7/M8 specs are unobservable under literal
  action 0.5 (equilibrium lock, ~3 spawn events total). Verified the same formulas directly at
  forced difficulty values instead; a future policy-based harness that actually climbs would let
  the literal checks work as written.
- The 0.5 equilibrium lock remains the dominant fact for any future training harness: a policy
  that idles at steer 0.5 gets H ≈ 123–193 and almost no spawns (M4 finding, unchanged by M7–M9:
  monsters at t≈0 have 0.0 spawn chance, breakables sit at the 0.05 base).

All milestones M1–M10 complete (M6 and M10 delivered as written+compiled code, not executed, due
to pygame being unavailable on this box). `env.py` is ready for headless NN training:
`reset()` returns 23 floats, `step([steer, shoot])` returns `(state, fitness, done)`, zero external
dependencies.

---

## UI polish pass (post-M10, demo only)

Goal: prettier, more polished UI; a "real" Doodle-Jump-style character; monsters that look like
actual monsters. Kept low-pixel-res/chunky on purpose.

**New `art.py`** — pure pixel-grid sprites (character = one solid-color cell, `.` = transparent),
scaled up with no smoothing so the art stays chunky.
- **Player** (12x12 grid @ scale 3 → 36x36): a yellow, round-bodied doodle jumper with big
  white/pupil eyes, a mouth, side shading, and orange feet. Reads as a little cartoon jumper, not
  a plain rounded square.
- **Two monster types** (9x9 grid @ scale 4 → 36x36 = exact monster hitbox):
  - **Grump** — a green slime/blob with white eyes + pupils, a fanged mouth, and little feet.
  - **Spiky** — a purple spiked blob with a spiky cap, angry eyes, and a mouth.
  Monsters alternate type by `index % 2` in the draw loop.
- `make_surface(grid, colors, scale)` is the single renderer; `player_sprite()` /
  `monster_sprites()` are the public factories.

**Rewrote `demo_render.py`** around the same env interface (unchanged `env.py`):
- Layered background: vertical gradient + ~50 deterministic twinkling stars + two rows of
  rounded "hill" ellipses for parallax-style depth.
- **Platforms** now have a two-tone top edge (lighter highlight + body) so they read as raised
  capsules; breakables get the amber palette plus a couple of darker crack strokes.
- **Bullets** = glowing rounded core with a soft glow (SRCALPHA pre-render), centered on the env
  bullet rect.
- **HUD** = pill-shaped translucent badges: live `H` (height) top-left, `BEST` top-right; a
  controls hint shown only on the opening screen.
- **Game-over** = centered rounded panel (translucent, purple border): title, final height +
  "NEW BEST!" or "Best …", and the R/Esc hint. Best score persists across runs in-session.
- Window icon set to the player sprite (guarded, so a headless icon failure can't crash startup).

**Verification this box allows** (pygame still not installable, PEP 668 — same as M6/M10):
- `python3 -m py_compile art.py demo_render.py` → OK.
- `python3 demo_render.py` headless → prints the normal skip message and exits 0 (guard intact).
- Sprite grids validated (row widths consistent); preview PNGs generated for inspection.
- `env.py` 3000-step smoke re-run (random steer) → state stayed 23 floats, loop terminated cleanly
  on a monster contact. **The polished look itself is still un-observed on screen** (no display,
  no pygame here) — correct by construction, same caveat as M10's visual confirmations.

Nothing in `env.py` or the env interface changed; this pass only touches `art.py` +
`demo_render.py`, so all headless self-checks/check scripts remain valid.

## Sound pass (post-M10, demo only)

First pass synthesized chiptune SFX in `sounds.py` from math at load — user
reported them bugged. Second pass: MoxieCat CC0 8-bit pack (OpenGameArt) —
user reported the bounce "too magical, too loud, drowned everything else".
Final pass (current): **authentic WAVs ripped from Doodle Jump Arcade**,
collected by LeDerpSillyGoober, hosted on The Sounds Resource (fan rip of
the 2009 game — demo-only assets, do not ship commercially):
https://sounds.spriters-resource.com/arcade/doodlejumparcade/asset/450387/

All files converted to 16-bit mono 44100 Hz, trimmed to a punchy active
region (5 ms head, 15 ms fade-out), and **normalized to the same -3 dBFS
peak** so the frequent bounce no longer drowns out rarer events:
bounce<-DJ_Jump (0.09 s plop), shoot<-rocket launch burst (0.35 s),
stomp<-jumponmonster (1.09 s), hit<-monster-crash (0.95 s),
crack<-explodingplatform (0.80 s), die<-Start_Failure (0.19 s).
`sounds.load()` builds the mixer bank from `sounds/*.wav` (returns {}
headless, verified); `demo_render.py` triggers by diffing env state: stomp
when a monster vanishes and feet rise, bounce otherwise, hit when a monster
vanishes without a rise (bullet), crack on platform removal, die on done.
env interface unchanged. WAV validity verified via `file` + RIFF parse,
all six distinct by md5. **Honest caveat:** audio was not heard on this box
(no pygame/display) — loudness balance is by construction (identical
peaks); perception judged on user's Mac.
