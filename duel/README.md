# Fly Duel

Two virtual flies with glowing blades, trained to duel by self-play with PPO: the same idea
as the viral clip, rebuilt from scratch. **No connectome involved**, like the original: the
controller is a small neural network (16 → 64 → 64 → 3: walk, turn, swing).

- `core.js`: the 2D arena (bodies, blades, parries, hits), a tiny MLP with hand-written
  backprop (gradient-checked) and PPO with GAE. Plain JS; runs in Node and the browser.
- `train.js`: `node duel/train.js 700 7` trains for 700 PPO updates (~1.4M steps, ~18 min on
  two CPU cores) and saves checkpoints to `checkpoints.json`.
- `build.py`: bakes the checkpoints into `fly-duel.html`.

Rewards: +1 for a hit, -1 for being hit, plus tiny nudges to face and close on the opponent.
An earlier version also rewarded parries and swinging near the opponent; the flies learned
to farm parries forever and never land a hit, so those went.
