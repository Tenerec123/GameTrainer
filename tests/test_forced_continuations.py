"""Tests for forced continuations (moves that do not change the turn)."""
from dataclasses import dataclass

import numpy as np
from gametrainer import Game, EncodableGame, MCTS, MatchMaker
from gametrainer.mcts import Node
from gametrainer.mcts import RolloutEvaluator


class ForcedTurnGame(Game[int, int]):
    """Position is an int cell; move is target cell = position + 1.

    If the target cell is even, the same player moves again (forced
    continuation); otherwise the turn alternates.
    """
    def __init__(self, length=8):
        self.length = length

    def get_initial_position(self):
        return 0  # cell 0, turn +1 (encoded as +cell? store turn separately)

    # We store (cell, turn) as a tuple instead; redefine below.
    # (placeholder, overridden by class below)


# Redefine cleanly with tuple positions.
@dataclass
class FTPos:
    cell: int
    turn: int


class FTGame(Game[FTPos, int]):
    def __init__(self, length=8):
        self.length = length

    def get_initial_position(self):
        return FTPos(0, 1)

    def get_legal_moves(self, position):
        if position.cell >= self.length - 1:
            return []
        return [position.cell + 1]

    def make_move(self, position, move):
        new_cell = move
        new_turn = position.turn if (new_cell % 2 == 0) else -position.turn
        return FTPos(new_cell, new_turn)

    def next_turn(self, position, move):
        new_cell = move
        return position.turn if (new_cell % 2 == 0) else -position.turn

    def is_game_over(self, position):
        return position.cell >= self.length - 1

    def get_result(self, position, player):
        if not self.is_game_over(position):
            return None
        # last mover (the one who reached the end) wins
        return 1.0 if (player == -position.turn) else -1.0

    def clone(self, position):
        return FTPos(position.cell, position.turn)


def test_next_turn_even_keeps_turn():
    g = FTGame()
    pos = FTPos(0, 1)
    assert g.next_turn(pos, 2) == 1


def test_next_turn_odd_alternates():
    g = FTGame()
    pos = FTPos(0, 1)
    assert g.next_turn(pos, 1) == -1


def test_mcts_forced_continuation_to_play():
    game = FTGame(length=6)
    ev = RolloutEvaluator(game)
    mcts = MCTS(game, ev, 20)
    root = Node(0, -1)
    root.visit_count = 1
    pos = FTPos(1, -1)
    mcts.run(root, pos)
    assert len(root.children) == 1
    child = list(root.children.values())[0]
    assert child.to_play == -1  # forced continuation: same turn


def test_mcts_alternating_to_play():
    game = FTGame(length=6)
    ev = RolloutEvaluator(game)
    mcts = MCTS(game, ev, 20)
    root = Node(0, 1)
    root.visit_count = 1
    pos = game.get_initial_position()
    mcts.run(root, pos)
    child = list(root.children.values())[0]
    assert child.to_play == -1


def test_selfplay_forced_continuation_sign():
    # Encodable wrapper
    class FTEnc(FTGame, EncodableGame):
        input_shape = (1, 1, 6)
        policy_shape = (1, 1, 6)
        def get_board_input(self, position):
            return np.zeros((1, 1, 6), dtype=np.float32)
        def get_move_mask(self, position):
            m = np.zeros((1, 1, 6), dtype=np.float32)
            for mv in self.get_legal_moves(position):
                m[0, 0, mv] = 1.0
            return m
        def get_move_probs(self, nn_output, to_play=1):
            arr = np.asarray(nn_output).reshape(-1)
            out = {}
            for i, p in enumerate(arr):
                if p > 0:
                    out[i] = float(p)
            return out
        def encode_policy_target(self, position, move_visits):
            t = np.zeros((1, 1, 6), dtype=np.float32)
            total = sum(move_visits.values()) or 1
            for mv, v in move_visits.items():
                t[0, 0, mv] = v / total
            return t

    game = FTEnc(length=6)
    ev = RolloutEvaluator(game)
    mm = MatchMaker(game, ev, max_moves=10, mcts_simulations=5)
    pos_list, policies, values = mm.run_match(n_step=2)
    # should complete without error
    assert isinstance(values, np.ndarray)
    # at least one position stored
    assert len(pos_list) >= 1
    # forced continuation: first move goes to cell1 (odd) -> alternates; no forced in this path necessarily, but no crash
    print("values", values)


def test_default_next_turn_on_base_game():
    # A game that never overrides next_turn alternates (TicTacToe-style)
    from games import TicTacToeCore
    g = TicTacToeCore()
    pos = g.get_initial_position()
    mv = g.get_legal_moves(pos)[0]
    assert g.next_turn(pos, mv) == -pos.turn
