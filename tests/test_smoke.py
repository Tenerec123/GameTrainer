"""End-to-end tests for the gametrainer core.

Uses a minimal TicTacToe implemented on the Game ABC plus a rollout
Evaluator, so the engine runs without any neural network.
"""
import random
from dataclasses import dataclass

import numpy as np

from gametrainer import Game, MCTS, MatchMaker
from gametrainer.mcts import Node

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]


def winner(board) -> int:
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


@dataclass(frozen=True)
class TicTacToePos:
    board: tuple  # 9 cells: 0 empty, +1 X, -1 O
    turn: int     # player to move

    def __str__(self):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [self.board[i:i + 3] for i in (0, 3, 6)]
        return "\n".join(" ".join(symbols[c] for c in row) for row in rows)


class TicTacToe(Game[TicTacToePos]):

    def get_initial_position(self) -> TicTacToePos:
        return TicTacToePos((0,) * 9, 1)

    def get_legal_moves(self, position) -> dict:
        return {i: [i] for i, cell in enumerate(position.board) if cell == 0}

    def make_move(self, position, move) -> TicTacToePos:
        origin, destiny = move
        if position.board[origin] != 0 or origin != destiny:
            raise ValueError(f"illegal move {move}")
        board = position.board[:origin] + (position.turn,) + position.board[origin + 1:]
        return TicTacToePos(board, -position.turn)

    def is_game_over(self, position) -> bool:
        return winner(position.board) != 0 or 0 not in position.board

    def get_result(self, position, player) -> float:
        w = winner(position.board)
        if w != 0:
            return float(w * player)
        if 0 not in position.board:
            return 0.0
        return None  # non-terminal

    def clone(self, position) -> TicTacToePos:
        return TicTacToePos(position.board, position.turn)

    # --- NN-facing members: stubs, not exercised by the rollout path ---

    def get_board_input(self, position):
        return np.array(position.board, dtype=np.float32)

    def get_move_mask(self, position):
        return np.array([1 if c == 0 else 0 for c in position.board], dtype=np.float32)

    def get_move_probs(self, nn_output, to_play=1) -> dict:
        raise NotImplementedError("NN path not exercised in these tests")

    def encode_policy_target(self, position, move_visits) -> np.ndarray:
        target = np.zeros(9, dtype=np.float32)
        total = sum(move_visits.values()) or 1
        for (origin, destiny), visits in move_visits.items():
            target[origin] = visits / total
        return target


class RolloutEvaluator:
    """Uniform priors + value from a random rollout to the end."""

    def __init__(self, game):
        self.game = game

    def _legal(self, position):
        return [(o, d) for o, ds in self.game.get_legal_moves(position).items() for d in ds]

    def evaluate(self, position):
        legal = self._legal(position)
        if not legal:
            return {}, 0.0
        priors = {move: 1.0 / len(legal) for move in legal}
        pos = position
        while not self.game.is_game_over(pos):
            pos = self.game.make_move(pos, random.choice(self._legal(pos)))
        value = self.game.get_result(pos, position.turn)
        return priors, float(value if value is not None else 0.0)


def test_mcts_standalone_picks_a_move():
    """The engine is playable on its own: search, pick by visits."""
    random.seed(7)
    np.random.seed(7)

    game = TicTacToe()
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


def test_matchmaker_generates_dataset():
    """Self-play produces aligned positions, policy targets and value targets."""
    random.seed(7)
    np.random.seed(7)

    game = TicTacToe()
    matchmaker = MatchMaker(game, RolloutEvaluator(game),
                            max_moves=12, mcts_simulations=60)
    positions, policies, values = matchmaker.run_match()

    assert len(positions) > 0
    assert len(positions) == len(policies) == len(values)
    assert policies.shape == (len(positions), 9)
    np.testing.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(values >= -1.0) and np.all(values <= 1.0)
    assert all(pos.turn in (1, -1) for pos in positions)
