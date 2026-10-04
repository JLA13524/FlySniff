# Fly Duel

Two flying flies with glowing blades, trained to duel by self-play with PPO: the same idea
as the viral clip, rebuilt from scratch. **No connectome involved**, like the original: the
controller is a small neural network (24 → 64 → 64 → 6: walk, turn, swing, climb, blade tilt,
sideways dash). The page shows both flies' networks firing live.

- `core.js`: the 2D arena (bodies, blades, parries, hits), a tiny MLP with hand-written
  backprop (gradient-checked) and PPO with GAE. Plain JS; runs in Node and the browser.
- `train.js`: `node duel/train.js 700 7` trains for 700 PPO updates (~1.4M steps, ~18 min on
  two CPU cores) and saves checkpoints to `checkpoints.json`.
- `build.py`: bakes the checkpoints into `fly-duel.html`.

Rewards: +1 for a hit, -1 for being hit, ±0.15 for a dodge (zero-sum), plus tiny nudges to face
and close on the opponent. Reward hacking found along the way, all fixed:
- rewarding parries: they parried each other 20 times a second forever and never hit;
- a non-zero-sum dodge bonus: they hovered at the ceiling trading near-misses;
- equal walk and fly speeds: they gave up flying, so walking is now slow and flying fast.

`train.js <iters> <seed> <from>` can continue from a saved checkpoint, which is how the rule
changes were trained in (from 200 and 400 updates).
