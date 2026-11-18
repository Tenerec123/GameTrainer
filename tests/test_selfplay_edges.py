"""Self-play edge cases."""
import random

import numpy as np

from gametrainer import MatchMaker
from games import RolloutEvaluator, TicTacToe


def _seed():
    random.seed(7)
    np.random.seed(7)


def test_matchmaker_max_moves_termination():
    _seed()
    game = TicTacToe()
    mm = MatchMaker(game, RolloutEvaluator(game), max_moves=4, mcts_simulations=20)
    positions, policies, values = mm.run_match()
    assert len(positions) <= 4
    assert len(positions) == len(policies) == len(values)
    assert np.all(values >= -1.0) and np.all(values <= 1.0)


def test_matchmaker_values_in_range_and_shapes():
    _seed()
    game = TicTacToe()
    mm = MatchMaker(game, RolloutEvaluator(game), max_moves=12, mcts_simulations=30)
    positions, policies, values = mm.run_match()
    assert policies.shape == (len(positions), 1, 3, 3)
    np.testing.assert_allclose(policies.sum(axis=(1, 2, 3)), 1.0, atol=1e-5)
    assert all(p.turn in (1, -1) for p in positions)


def test_matchmaker_n_step_param_accepted():
    _seed()
    game = TicTacToe()
    mm = MatchMaker(game, RolloutEvaluator(game), max_moves=8, mcts_simulations=20)
    positions, policies, values = mm.run_match(n_step=2)
    assert len(positions) > 0


def test_matchmaker_empty_game_returns_empty():
    # A game already terminal at initial position
    class TerminalGame(TicTacToe):
        def is_game_over(self, position):
            return True

        def get_result(self, position, player):
            return 0.0

        def get_legal_moves(self, position):
            return {}

    _seed()
    game = TerminalGame()
    mm = MatchMaker(game, RolloutEvaluator(game), max_moves=5, mcts_simulations=5)
    positions, policies, values = mm.run_match()
    assert positions == [] or len(positions) == 0
    assert np.asarray(policies).size == 0
    assert np.asarray(values).size == 0
