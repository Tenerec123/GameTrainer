"""MCTS edge cases."""
import random

import numpy as np

from gametrainer import MCTS
from gametrainer.mcts import Node, root_noise
from games import RolloutEvaluator, TicTacToe, TicTacToeCore


def _seed():
    random.seed(7)
    np.random.seed(7)


def test_mcts_num_simulations_zero():
    _seed()
    game = TicTacToeCore()
    ev = RolloutEvaluator(game)
    mcts = MCTS(game, ev, num_simulations=0)
    root = Node(0, 1)
    mcts.run(root, game.get_initial_position())
    # root may or may not be expanded; no crash
    assert root.visit_count >= 0


def test_mcts_terminal_root_no_crash():
    _seed()
    game = TicTacToeCore()
    ev = RolloutEvaluator(game)
    # build a terminal position: full board draw
    board = (1, -1, 1, -1, 1, -1, -1, 1, -1)
    pos = TicTacToeCore.__wrapped__ if hasattr(TicTacToeCore, "__wrapped__") else None
    from games import TicTacToePos
    tpos = TicTacToePos(board, 1)
    assert game.is_game_over(tpos)
    mcts = MCTS(game, ev, num_simulations=20)
    root = Node(0, 1)
    mcts.run(root, tpos)  # must not crash
    assert True


def test_mcts_already_expanded_root_reused():
    _seed()
    game = TicTacToeCore()
    ev = RolloutEvaluator(game)
    mcts = MCTS(game, ev, num_simulations=10)
    root = Node(0, 1)
    mcts.run(root, game.get_initial_position())
    n_children = len(root.children)
    # second run on same root should not crash or wipe children
    mcts.run(root, game.get_initial_position())
    assert len(root.children) == n_children


def test_root_noise_single_child_noop():
    game = TicTacToeCore()
    root = Node(0, 1)
    # manually expand with one child
    from gametrainer.mcts import Node as N
    root.children = {"m": N(0.5, -1)}
    root_noise(root)  # should not crash, n_actions<=1 returns
    assert root.children["m"].prior == 0.5


def test_mcts_empty_legal_moves_game():
    # A game whose initial position has no legal moves
    class NoMoves(TicTacToeCore):
        def get_legal_moves(self, position):
            return {}

        def is_game_over(self, position):
            return True

        def get_result(self, position, player):
            return 0.0

    _seed()
    game = NoMoves()
    ev = RolloutEvaluator(game)
    mcts = MCTS(game, ev, num_simulations=10)
    root = Node(0, 1)
    mcts.run(root, game.get_initial_position())  # no crash
    assert len(root.children) == 0
