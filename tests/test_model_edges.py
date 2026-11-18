"""Model/evaluator edge cases."""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gametrainer import NeuralEvaluator, build_net
from games import TicTacToe


def test_evaluator_terminal_position_no_crash():
    game = TicTacToe()
    torch.manual_seed(0)
    ev = NeuralEvaluator(game, arch="tiny")
    # full board draw
    board = (1, -1, 1, -1, 1, -1, -1, 1, -1)
    from games import TicTacToePos
    pos = TicTacToePos(board, 1)
    priors, value = ev.evaluate(pos)
    assert isinstance(priors, dict)
    assert -1.0 <= value <= 1.0


def test_evaluator_respects_device_cpu():
    game = TicTacToe()
    ev = NeuralEvaluator(game, arch="tiny", device="cpu")
    assert str(ev.device) == "cpu"
    priors, value = ev.evaluate(game.get_initial_position())
    assert abs(sum(priors.values()) - 1.0) < 1e-5


def test_build_net_rejects_spatial_mismatch():
    class BadGame(TicTacToe):
        policy_shape = (1, 2, 2)  # spatial mismatch with input 3x3

    with pytest.raises(ValueError):
        build_net(BadGame())


def test_custom_net_value_shape_1x1_accepted():
    game = TicTacToe()

    class Net(torch.nn.Module):
        def forward(self, x):
            n = x.shape[0]
            return torch.zeros(n, 1, 3, 3), torch.zeros(n, 1)

    ev = NeuralEvaluator(game, net=Net())
    priors, value = ev.evaluate(game.get_initial_position())
    assert isinstance(value, float)


def test_evaluator_no_warning_without_arch():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        game = TicTacToe()
        ev = NeuralEvaluator(game, net=_Net())
    assert ev is not None


class _Net(torch.nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return torch.zeros(n, 1, 3, 3), torch.zeros(n)
