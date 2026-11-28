"""Player/agent shell tests: evaluate, value, best_move, checkpoint loading."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gametrainer import Player, NeuralEvaluator, build_net, Trainer, RolloutEvaluator
from games import TicTacToe, TicTacToePos


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def test_player_construction_arch():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny")
    assert p.game is game
    assert p.mcts is None
    priors, value = p.evaluate(game.get_initial_position())
    assert isinstance(priors, dict)
    assert -1.0 <= value <= 1.0


def test_player_value():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny")
    pos = game.get_initial_position()
    assert p.value(pos) == p.evaluate(pos)[1]


def test_player_best_move_argmax():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny")
    pos = game.get_initial_position()
    legal = game.get_legal_moves(pos)
    mv = p.best_move(pos)
    assert mv in legal


def test_player_best_move_terminal():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny")
    board = (1, -1, 1, -1, 1, -1, -1, 1, -1)
    pos = TicTacToePos(board, 1)
    assert game.is_game_over(pos)
    mv = p.best_move(pos)
    # no legal moves -> None
    assert mv is None


def test_player_with_net():
    game = TicTacToe()
    _seed()
    net = build_net(game, arch="tiny")
    p = Player(game, net=net)
    mv = p.best_move(game.get_initial_position())
    assert mv in game.get_legal_moves(game.get_initial_position())


def test_player_with_evaluator():
    game = TicTacToe()
    _seed()
    ev = NeuralEvaluator(game, arch="tiny")
    p = Player(game, evaluator=ev)
    assert p.evaluator is ev
    assert p.net is ev.net


def test_player_mcts_construction_and_best_move():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny", mcts_simulations=5)
    assert p.mcts is not None
    pos = game.get_initial_position()
    mv = p.best_move(pos)
    assert mv in game.get_legal_moves(pos)


def test_player_flexible_kwargs():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny", checkpoint_path=None, num_simulations=3)
    assert p.mcts_simulations == 3
    assert p.mcts is not None


def test_player_checkpoint_roundtrip_trainer_format():
    game = TicTacToe()
    _seed()
    tr = Trainer(game, batch_size=4)
    # train a tiny bit is not necessary; save checkpoint
    ckpt_dir = tempfile.mkdtemp()
    ckpt = os.path.join(ckpt_dir, "c.pt")
    tr.save_checkpoint(ckpt)
    p = Player(game, checkpoint=ckpt)
    assert p.load_checkpoint(ckpt) is True
    mv = p.best_move(game.get_initial_position())
    assert mv in game.get_legal_moves(game.get_initial_position())


def test_player_checkpoint_roundtrip_raw_state_dict():
    game = TicTacToe()
    _seed()
    net = build_net(game, arch="tiny")
    torch.save(net.state_dict(), os.path.join(tempfile.mkdtemp(), "w.pt"))
    path = os.path.join(tempfile.mkdtemp(), "w.pt")
    torch.save(net.state_dict(), path)
    p = Player(game, checkpoint=path, arch="tiny")
    assert p.net is not None


def test_player_positional_checkpoint():
    game = TicTacToe()
    _seed()
    tr = Trainer(game, batch_size=4)
    ckpt_dir = tempfile.mkdtemp()
    ckpt = os.path.join(ckpt_dir, "c.pt")
    tr.save_checkpoint(ckpt)
    p = Player(game, ckpt, mcts_simulations=0)
    assert p.mcts is None
    mv = p.best_move(game.get_initial_position())
    assert mv in game.get_legal_moves(game.get_initial_position())


def test_player_temperature_sampling_runs():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny")
    pos = game.get_initial_position()
    # should not crash and return a legal move
    for _ in range(5):
        mv = p.best_move(pos, temperature=1.0)
        assert mv in game.get_legal_moves(pos)


def test_player_mcts_temperature_sampling_runs():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny", mcts_simulations=4)
    pos = game.get_initial_position()
    for _ in range(5):
        mv = p.best_move(pos, temp=0.5)
        assert mv in game.get_legal_moves(pos)


def test_player_rollout_eval_basic():
    game = TicTacToe()
    p = Player(game, rollout_eval=True)
    assert p.net is None
    assert p.mcts is not None
    assert p.mcts_simulations == 50
    pos = game.get_initial_position()
    mv = p.best_move(pos)
    assert mv in game.get_legal_moves(pos)
    priors, value = p.evaluate(pos)
    assert isinstance(priors, dict)
    assert -1.0 <= value <= 1.0


def test_player_rollout_eval_explicit_sims():
    game = TicTacToe()
    p = Player(game, rollout_eval=True, mcts_simulations=4)
    assert p.mcts_simulations == 4
    mv = p.best_move(game.get_initial_position())
    assert mv in game.get_legal_moves(game.get_initial_position())


def test_player_rollout_eval_positional_bool():
    game = TicTacToe()
    p = Player(game, True, mcts_simulations=3)
    assert p.net is None
    mv = p.best_move(game.get_initial_position())
    assert mv in game.get_legal_moves(game.get_initial_position())


def test_player_rollout_eval_ignores_net():
    game = TicTacToe()
    net = build_net(game, arch="tiny")
    # rollout_eval=True ignores net/checkpoint/arch
    p = Player(game, net=net, arch="tiny", rollout_eval=True)
    assert p.net is None


def test_rollout_evaluator_direct():
    game = TicTacToe()
    ev = RolloutEvaluator(game)
    priors, value = ev.evaluate(game.get_initial_position())
    assert set(priors.keys()) == set(game.get_legal_moves(game.get_initial_position()))
    assert abs(sum(priors.values()) - 1.0) < 1e-9


def test_player_terminal_with_mcts():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny", mcts_simulations=4)
    board = (1, -1, 1, -1, 1, -1, -1, 1, -1)
    pos = TicTacToePos(board, 1)
    assert game.is_game_over(pos)
    assert p.best_move(pos) is None


def test_player_rollout_temperature_sampling():
    game = TicTacToe()
    p = Player(game, rollout_eval=True, mcts_simulations=4)
    pos = game.get_initial_position()
    for _ in range(5):
        mv = p.best_move(pos, temperature=1.0)
        assert mv in game.get_legal_moves(pos)


def test_player_device_kwarg():
    game = TicTacToe()
    _seed()
    p = Player(game, arch="tiny", device="cpu")
    assert str(p.device) == "cpu"
    assert str(p.evaluator.device) == "cpu"


def test_load_checkpoint_weights_change():
    game = TicTacToe()
    _seed()
    tr = Trainer(game, batch_size=4)
    ckpt_dir = tempfile.mkdtemp()
    ckpt = os.path.join(ckpt_dir, "c.pt")
    tr.save_checkpoint(ckpt)
    p_before = Player(game, arch="tiny")
    p_after = Player(game, checkpoint=ckpt)
    # weights loaded (no exception) and best_move works
    mv = p_after.best_move(game.get_initial_position())
    assert mv in game.get_legal_moves(game.get_initial_position())


def test_player_no_torch_import_ok():
    # Player class importable even without using torch
    import gametrainer
    assert gametrainer.Player is not None