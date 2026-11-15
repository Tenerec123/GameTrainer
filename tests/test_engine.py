"""Engine tests: MCTS over the rules-only Game face.

Proves the segregation guarantee: the engine is playable on a Game with
zero neural-network members.
"""
import random

import numpy as np

from gametrainer import Game, MCTS
from gametrainer.mcts import Node
from games import RolloutEvaluator, TicTacToeCore


def test_mcts_standalone_picks_a_move():
    """The engine is playable on its own: search, pick by visits."""
    random.seed(7)
    np.random.seed(7)

    game = TicTacToeCore()
    assert isinstance(game, Game)
    # Segregation guarantee: the core face knows nothing about matrices
    assert not hasattr(game, "get_board_input")
    assert not hasattr(game, "get_move_mask")
    assert not hasattr(game, "get_move_probs")

    mcts = MCTS(game, RolloutEvaluator(game), num_simulations=200)
    root = Node(0, 1)
    mcts.run(root, game.get_initial_position())

    assert root.children, "root should be expanded after run()"
    best_move, best_child = max(root.children.items(),
                                key=lambda kv: kv[1].visit_count)
    assert best_child.visit_count > 0
    assert -1.0 <= root.value() <= 1.0
    legal = [(o, d) for o, ds in game.get_legal_moves(game.get_initial_position()).items()
             for d in ds]
    assert best_move in legal
