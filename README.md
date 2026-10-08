# SprudelJump

A small Doodle Jump clone, built as a training ground for a neural network that learns to play it.

- **`env.py`**: the headless game. Gym-style `reset()` / `step(action)`, no rendering, standard library only, so thousands of games can run in parallel.
- **`fastEnv.py`**: a numba-compiled port of the env that steps many games in lockstep. Same seed and same actions give the same scores bit for bit; `test_fastEnv.py` checks this frame by frame.
- **`demo_render.py`**: a pygame window around the same env. Play it yourself, or watch a trained network play.

## Run it

Python 3.10+.

```bash
pip install pygame
python demo_render.py                      # play: ←/→ or A/D to steer, Space to shoot, R to restart
pip install torch
python demo_render.py --model best.pt      # watch a trained network play (--stable / --smoothing to match its training)
pip install numpy numba
python test_fastEnv.py                     # fastEnv vs env.py, frame by frame
```

## The environment

- **Action:** `[steer, shoot]`, two floats in `[0, 1]`, straight from a 2-output sigmoid network. `steer` 0 = full left, 0.5 = still, 1 = full right. `shoot` > 0.5 fires.
- **Observation:** 23 floats by default, pre-normalised. Player x and vertical speed, the 5 nearest platforms (relative x, wrap-aware; relative y; breakable) and the 3 nearest monsters, all relative to the player's feet. Options: fixed-meaning slots instead of nearest-first (`stableSlots`), a difficulty input, per-platform landing prediction.
- **Fitness:** height climbed plus a bonus per monster shot.
- **Episode ends** on falling off the screen, touching a monster, or 600 frames without reaching a new height (stops agents bouncing in place forever).
- **Difficulty** ramps with height: wider gaps, narrower platforms, more breakable platforms, more monsters.
- **Seeded:** each env has its own RNG. Same seed = same level, so networks can be compared fairly. `reset()` options for practice: start higher up (`minStartHeight` / `maxStartHeight`), more monsters (`monsterFraction` / `monsterMult`).

## The learning side

The networks are trained by **neuroevolution**: a population of networks plays, the best are kept and mutated (optionally crossed over), repeat. The network and training loop live in [tensorNetwork](https://github.com/JoblessJoe/tensorNetwork), my learning project, written from scratch on PyTorch tensors to learn how a network actually works.

Training runs on my home rig: RTX 2080 Ti, Ryzen 9 7950X, 32 GB DDR5. Games run in parallel across the CPU cores.
