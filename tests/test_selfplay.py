"""Self-play tests: MatchMaker over the full EncodableGame face.

MatchMaker produces training data (canonical policy targets), so it sits
on the encodable side of the dependency map — but needs no torch.
"""
import random

import numpy as np

from gametrainer import EncodableGame, MatchMaker
from games import RolloutEvaluator, TicTacToe


def test_matchmaker_generates_dataset():
    """Self-play produces aligned positions, policy targets and value targets."""
    random.seed(7)
    np.random.seed(7)

    game = TicTacToe()
    assert isinstance(game, EncodableGame)

    matchmaker = MatchMaker(game, RolloutEvaluator(game),
                            max_moves=12, mcts_simulations=60)
    positions, policies, values = matchmaker.run_match()

    assert len(positions) > 0
    assert len(positions) == len(policies) == len(values)
    assert policies.shape == (len(positions), 1, 3, 3)
    np.testing.assert_allclose(policies.sum(axis=(1, 2, 3)), 1.0, atol=1e-5)
    assert np.all(values >= -1.0) and np.all(values <= 1.0)
    assert all(pos.turn in (1, -1) for pos in positions)
