# SprudelJump — STATUS

## Built

`env.py` — headless Doodle-Jump environment, pure stdlib (only `import random`), no pygame.
Complete per CORE_SPEC interface: `reset() -> list[12 floats]`,
`step(action in [0,1]) -> (state, fitness, done)`, `render() = pass`.
- Physics: gravity 0.4, bounce -13.0, max fall 15.0, horizontal speed 6.0, screen wrap.
- Camera scroll at playerY < 280 (0.4*700); platforms shift down, despawn below screen, spawn above top.
- Difficulty scaling t = totalHeight/20000: gap 80–130 → 100–160, platform width 60 → 40.
- State: playerX/W, clamped velY, then 5 nearest above-platforms as (dx/W clamped, dy/H clamped); padding [0.0, 1.0].
- Fitness = running totalHeight (score); done when player falls off bottom.

`demo_render.py` — optional pygame viewer (player/platforms as rects, width per platform,
H drawn on screen, keyboard play + auto-random, Esc to quit, R to reset). Under 100 lines.
pygame is NOT importable from env.py.

Milestone checks: `check_m2.py`, `check_m3.py`, `check_m4.py`, `check_m5.py` (M1 checked inline).

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
