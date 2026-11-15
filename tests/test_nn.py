"""Neural tests: NeuralEvaluator + generic net over the EncodableGame face.

These need torch (the [nn] extra). The module is skipped gracefully
where torch is absent so the rest of the suite stays green anywhere.
"""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# torch is guaranteed from here on
from gametrainer import MCTS, NeuralEvaluator, build_net
from gametrainer.mcts import Node
from games import TicTacToe


class FixedNet(torch.nn.Module):
    """Deterministic custom net: zero logits, zero value."""

    def forward(self, x):
        n = x.shape[0]
        return torch.zeros(n, 1, 3, 3), torch.zeros(n)


def test_build_net_forward_shapes():
    game = TicTacToe()
    torch.manual_seed(0)
    net = build_net(game, arch="tiny")

    x = torch.randn(4, *game.input_shape)
    logits, value = net(x)

    assert logits.shape == (4, *game.policy_shape)
    assert value.shape == (4,)
    assert bool((value >= -1.0).all()) and bool((value <= 1.0).all())


def test_build_net_rejects_unknown_arch():
    with pytest.raises(ValueError, match="unknown arch"):
        build_net(TicTacToe(), arch="huge")


def test_probe_rejects_lie_about_input_shape():
    class Liar(TicTacToe):
        input_shape = (3, 3, 3)  # declared shape does not match get_board_input

    with pytest.raises(ValueError, match="input_shape"):
        NeuralEvaluator(Liar())


def test_probe_rejects_wrong_mask_shape():
    class BadMask(TicTacToe):
        def get_move_mask(self, position):
            return np.zeros(9, dtype=np.float32)  # flat, not (1, 3, 3)

    with pytest.raises(ValueError, match="policy_shape"):
        NeuralEvaluator(BadMask())


def test_evaluate_returns_normalized_priors():
    game = TicTacToe()
    torch.manual_seed(0)
    ev = NeuralEvaluator(game, arch="tiny")

    priors, value = ev.evaluate(game.get_initial_position())

    legal = {(i, i) for i, cell in enumerate(game.get_initial_position().board) if cell == 0}
    assert set(priors) == legal
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert all(0.0 < p <= 1.0 for p in priors.values())
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_evaluate_respects_legality_after_moves():
    game = TicTacToe()
    torch.manual_seed(0)
    ev = NeuralEvaluator(game, arch="tiny")

    pos = game.make_move(game.get_initial_position(), (0, 0))
    pos = game.make_move(pos, (4, 4))
    priors, _ = ev.evaluate(pos)

    legal = {(i, i) for i, cell in enumerate(pos.board) if cell == 0}
    assert set(priors) == legal
    assert abs(sum(priors.values()) - 1.0) < 1e-5


def test_evaluate_with_custom_net_is_deterministic():
    game = TicTacToe()
    ev = NeuralEvaluator(game, net=FixedNet())

    priors, value = ev.evaluate(game.get_initial_position())

    assert len(priors) == 9
    np.testing.assert_allclose(sorted(priors.values()), [1 / 9] * 9, atol=1e-6)
    assert value == 0.0


def test_custom_net_warns_when_arch_is_also_given():
    with pytest.warns(UserWarning, match="ignored"):
        NeuralEvaluator(TicTacToe(), arch="tiny", net=FixedNet())


def test_custom_net_warns_even_for_explicit_default_arch():
    with pytest.warns(UserWarning, match="ignored"):
        NeuralEvaluator(TicTacToe(), arch="small", net=FixedNet())


def test_custom_net_without_arch_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ev = NeuralEvaluator(TicTacToe(), net=FixedNet())

    priors, _ = ev.evaluate(TicTacToe().get_initial_position())
    assert len(priors) == 9


def test_mcts_runs_with_neural_evaluator():
    """The learned evaluator speaks the Evaluator protocol: MCTS just works."""
    game = TicTacToe()
    torch.manual_seed(0)
    ev = NeuralEvaluator(game, arch="tiny")

    mcts = MCTS(game, ev, num_simulations=25)
    root = Node(0, 1)
    mcts.run(root, game.get_initial_position())

    assert root.children, "root should be expanded"
    best_move, best_child = max(root.children.items(), key=lambda kv: kv[1].visit_count)
    assert best_child.visit_count > 0
    assert -1.0 <= root.value() <= 1.0
    legal = {(o, d) for o, ds in game.get_legal_moves(game.get_initial_position()).items()
             for d in ds}
    assert best_move in legal
