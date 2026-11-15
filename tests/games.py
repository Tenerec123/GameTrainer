"""Shared game fixtures for the gametrainer test suite.

Mirrors the architecture: TicTacToeCore is the rules-only Game face;
TicTacToe adds the EncodableGame face on top (the same inheritance
pattern as the ABCs themselves).
"""
import random
from dataclasses import dataclass

import numpy as np

from gametrainer import EncodableGame, Game

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


class TicTacToeCore(Game[TicTacToePos]):
    """Rules-only face: zero neural-network members."""

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


class TicTacToe(TicTacToeCore, EncodableGame[TicTacToePos]):
    """Full contract: rules + canonical-space (NN) face."""

    # Canonical space (channels-first, no batch dim)
    input_shape = (2, 3, 3)   # my pieces, opponent pieces (mover's view)
    policy_shape = (1, 3, 3)  # one channel: the 3x3 grid of cells

    def get_board_input(self, position):
        board = np.asarray(position.board, dtype=np.float32).reshape(3, 3)
        return np.stack([
            (board == position.turn).astype(np.float32),
            (board == -position.turn).astype(np.float32),
        ])

    def get_move_mask(self, position):
        empty = np.array([1 if c == 0 else 0 for c in position.board], dtype=np.float32)
        return empty.reshape(1, 3, 3)

    def get_move_probs(self, nn_output, to_play=1) -> dict:
        # The evaluator delivers the policy already masked and normalized:
        # illegal cells carry exactly 0.0, so p > 0 identifies legal moves.
        probs = np.asarray(nn_output, dtype=np.float32).reshape(-1)
        return {(i, i): float(p) for i, p in enumerate(probs) if p > 0.0}

    def encode_policy_target(self, position, move_visits) -> np.ndarray:
        target = np.zeros((1, 3, 3), dtype=np.float32)
        total = sum(move_visits.values()) or 1
        for (origin, destiny), visits in move_visits.items():
            target[0, origin // 3, origin % 3] = visits / total
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
