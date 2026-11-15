from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np
Position = TypeVar("Position")


class Game(ABC, Generic[Position]):
    """Base game interface for MCTS + AlphaZero architecture.

    The Game is a pure engine: it receives positions and returns results.
    It has no internal state — turn tracking lives in the MCTS Node.

    Position is the data structure that defines the game state.
    It should expose the player to move (attribute `turn`: +1/-1) and
    a __str__ for logs.

    This contract is rules-only: no neural-network knowledge lives here.
    The canonical-space (NN) face is the EncodableGame contract.
    """

    @abstractmethod
    def get_initial_position(self) -> Position:
        """Return the starting point of the game"""
        ...

    @abstractmethod
    def get_legal_moves(self, position: Position) -> dict:
        """Return all legal moves from the given position.

        Returns a dict: {origin: [destiny, ...]}
        """
        ...

    @abstractmethod
    def make_move(self, position: Position, move) -> Position:
        """Apply a move and return a NEW position (no mutation)."""
        ...

    @abstractmethod
    def is_game_over(self, position: Position) -> bool:
        """Check if the game has ended."""
        ...

    @abstractmethod
    def get_result(self, position: Position, player) -> float:
        """Return the result for `player` from the given position.

        Returns:
            +1.0  win
            -1.0  loss
             0.0  draw
        """
        ...

    @abstractmethod
    def clone(self, position: Position) -> Position:
        """Return a deep copy of the position."""
        ...

    def __str__(self):
        return super().__str__()


class EncodableGame(Game[Position]):
    """Neural-network face of a game

    Only required on the NN path.
    The network always sees the board from the perspective of the player to move (canonical space).
    This contract encodes positions into that space and decodes policies back into real moves.
    All shapes are channels-first (C, H, W) with no batch dimension.
    """

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, int, int]:
        """(C, H, W) returned by get_board_input."""
        ...

    @property
    @abstractmethod
    def policy_shape(self) -> tuple[int, int, int]:
        """(C_pol, H, W) of get_move_mask, encode_policy_target and the
        policy the evaluator hands to get_move_probs."""
        ...

    @abstractmethod
    def get_board_input(self, position: Position) -> np.ndarray:
        """Encode the position as a float32 array with shape input_shape,
        in canonical space (player-to-move perspective)."""
        ...

    @abstractmethod
    def get_move_mask(self, position: Position) -> np.ndarray:
        """Return a float32 mask of legal moves with shape policy_shape
        (1.0 legal, 0.0 illegal)."""
        ...

    @abstractmethod
    def get_move_probs(self, nn_output, to_play=1) -> dict:
        """Receives the canonical policy (numpy, policy_shape, already
        masked and normalized) and outputs a dict with key -> move and
        value -> policy percentage (0-1).

        to_play: 1 for white, -1 for black. Used to un-flip canonical coordinates.
        """
        ...

    @abstractmethod
    def encode_policy_target(self, position: Position, move_visits: dict) -> np.ndarray:
        """Encode a visit-count distribution over real moves as a policy
        target, in the SAME canonical space the network outputs (matching
        get_board_input/get_move_mask orientation for the player to move)."""
        ...

    @staticmethod
    def encode_move(o_x, o_y, n_x, n_y):
        """Encode a move (o_x, o_y, n_x, n_y) to (o_x, o_y, channel).

        Returns a tuple matching _decode_move's output shape: (o_x, o_y, channel).
        Each game implementation defines its own channel mapping.
        """
        raise NotImplementedError("Subclass must implement encode_move")
