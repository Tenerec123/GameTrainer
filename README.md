# GameTrainer
A library to create and train an AI to play games such us Chess, Checkers, Go or Strategeti. It implements the MCTS, the model creation and the trainloop.

The only work it requires is to implement the game engine, which includes the position class, the conversion of that position into parameters, the conversion of model output into moves and the decider of which moves are legal and which ones are not.

install it with: pip install git+https://github.com/tenerec123/GameTrainer.git

## Playing with a trained model

Once a model is trained (or to use a zero-training baseline), wrap it in a
`Player`:

```python
from gametrainer import Player

# Neural (requires gametrainer[nn]):
player = Player(game, checkpoint="checkpoint.pt", mcts_simulations=25)

# Pure MCTS with a rollout evaluator (no torch):
player = Player(game, rollout_eval=True, mcts_simulations=25)

pos = game.get_initial_position()
while not game.is_game_over(pos):
    move = player.best_move(pos)            # greedy; temperature= for exploration
    pos = game.make_move(pos, move)

priors, value = player.evaluate(pos)        # single forward pass
```

`Player` exposes `evaluate`, `value`, `best_move` and `load_checkpoint`.

## Implementing a game

Subclass `Game` (rules only) or `EncodableGame` (rules + neural-network face).
A move is free-form and must be hashable. `get_legal_moves` returns a `list[Move]`.

By default every move alternates the turn (`next_turn` returns `-position.turn`).
For games with forced continuations (a move that does not change the turn),
override `next_turn(position, move)` (single) or `next_turns(position, moves)`
(batch) to return the player to move afterwards. MCTS and self-play use this
to keep `to_play` consistent. Positions with a single legal move are fast-pathed
in MCTS/Player and are not stored as training examples in self-play.