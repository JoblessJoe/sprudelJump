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
python demo_render.py --model best.pt      # watch a trained network play; defaults match the current nets (--repeat 5 --smoothing 0.96875)
python demo_render.py --model best.pt --speed 4   # four game frames per picture (sound off above 1)
pip install numpy numba
python test_fastEnv.py                     # fastEnv vs env.py, frame by frame
```

## The environment

- **Action:** `[steer, shoot]`, two floats in `[0, 1]`, straight from a 2-output sigmoid network. `steer` 0 = full left, 0.5 = still, 1 = full right. `shoot` > 0.5 fires.
- **Observation:** 33 floats, pre-normalised, every slot with a fixed meaning. Player x and vertical speed; 5 platform slots (3 nearest below the feet, 2 nearest above: relative x, wrap-aware; relative y; breakable); 3 monster slots (1 below, 2 above: relative x, relative y); and a landing prediction per platform slot (frames until the player lands on it, how much steering that needs). All relative to the player's feet. The layout is set by the constants at the top of `env.py` (`PLATFORMS_BELOW`, ...); `_getState` documents it.
- **Fitness:** height climbed plus a bonus per monster shot. A bullet only kills a monster that is at least partly on the screen; monsters above the top edge cannot be hit.
- **Episode ends** on falling off the screen, touching a monster, or 600 frames without reaching a new height (stops agents bouncing in place forever).
- **Difficulty** ramps with height: wider gaps, narrower platforms, more breakable platforms, more monsters.
- **Seeded:** each env has its own RNG. Same seed = same level, so networks can be compared fairly. `reset()` options for practice: start higher up (`minStartHeight` / `maxStartHeight`), more monsters (`monsterFraction` / `monsterMult`).

## The learning side

The networks are trained by **neuroevolution**: a population of networks plays, the best are kept and mutated (optionally crossed over), repeat. The network and training loop live in [tensorNetwork](https://github.com/JoblessJoe/tensorNetwork), my learning project, written from scratch on PyTorch tensors to learn how a network actually works.

Training runs on my home rig: RTX 2080 Ti, Ryzen 9 7950X, 32 GB DDR5. Games run in parallel across the CPU cores.
