from .game import Game
import numpy as np
from .mcts import MCTS, Node

class MatchMaker:
    def __init__(self, game:Game, evaluator, max_moves, mcts_simulations):
        self.game = game
        self.evaluator = evaluator
        self.max_moves = max_moves
        self.mcts_simulations = mcts_simulations
        self.mcts = MCTS(self.game, self.evaluator, self.mcts_simulations)

    def _print_move_traceback(self, pos_list):
        """Print the full move list with board after each position."""
        print(f"\n{'='*60}")
        print(f"  MOVE TRACEBACK — {len(pos_list)} positions")
        print(f"{'='*60}")
        for i, item in enumerate(pos_list, start=1):
            print()
            print(i)
            print(item)
            print()
        print(f"{'='*60}\n")

    def run_match(self, n_step=5):
        """Run a self-play match, collecting position, policy, and value data.

        For terminal wins/losses, value targets are the actual game result
        (±1 from each player's perspective). For true draws (no legal moves
        remain, e.g. stalemate), the target is 0.
        For non-terminal games and adjudicated draws (the game engine ends
        the game while legal moves remain, e.g. chess's rule50 counter),
        value targets use n-step MCTS lookahead: position k's target = MCTS
        root value at position k+n (converted to position k's player
        perspective). If k+n exceeds the game length, uses the last
        available position's MCTS value.
        An adjudicated draw is a property of the continuation played, not
        of the position, so it gets the same n-step treatment as a
        non-terminal game.

        Sign logic (critical):
          MCTS values are from the perspective of the player TO MOVE at that
          position. To convert from position j's perspective to position k's:
            target_k = v_j * turn_k * turn_j
          where turn is the player to move (+1/-1).
        """
        current_pos = self.game.get_initial_position()
        current_node = Node(0, 1)
        current_node.visit_count = 1
        pos_list = []
        mcts_values = []  # (root_value, to_play) per stored position
        policy_targets = []
        for turn in range(self.max_moves):
            temp = 1 if turn <= 60 else 0.2
            if self.game.is_game_over(current_pos): 
                break

            result = self.mcts.run(current_node, current_pos)
            if result is not None:
                self._print_move_traceback(pos_list+result)
                raise RuntimeError("El rey ha desaparecido, aquí el traceback ^")
            
            visit_counts = {move: c.visit_count for move, c in current_node.children.items()}
            policy_targets.append(self.game.encode_policy_target(current_pos, visit_counts))

            weights = np.array([c.visit_count ** (1 / temp) for c in current_node.children.values()])
            weights /= weights.sum()          # normaliza por la suma REAL: robusto siempre
            next_move_idx = np.random.choice(len(weights), p=weights)

            next_move = list(current_node.children.keys())[next_move_idx]
            next_node = current_node.children[next_move]
            pos_list.append(current_pos)
            mcts_values.append(current_node.value())
            current_pos = self.game.make_move(current_pos, next_move)

            del current_node.children[next_move]
            current_node.children.clear()
            del current_node
            current_node = next_node

        moves_played = len(pos_list)
        if moves_played == 0:
            return [], np.array([]), np.array([])

        last_value_white = None
        is_fifty_move_draw = False
        if self.game.is_game_over(current_pos):
            last_value_white = self.game.get_result(current_pos, 1)
            # A terminal draw that still has legal moves is an adjudicated
            # (prolongation/repetition) draw: it says nothing about the value
            # of the positions — the draw is a property of the continuation
            # played, not of the board. Route it through the same n-step
            # bootstrap as non-terminal games so the value gradient survives.
            # WHEN such a draw fires is the game engine's business (chess:
            # the rule50 counter lives in the position metadata and the
            # engine ends the game when it triggers).
            # True draws (e.g. stalemate) have no legal moves left: target 0.
            is_adjudicated_draw = (
                last_value_white == 0.0
                and len(self.game.get_legal_moves(current_pos)) > 0
            )

        if last_value_white is not None and last_value_white != 0.0:
            value_targets = np.array([last_value_white * pos.turn for pos in pos_list])
            if last_value_white == 1.0:
                result = f"White wins in {moves_played} moves"
            else:
                result = f"Black wins in {moves_played} moves"
        elif is_adjudicated_draw or last_value_white is None:
            value_targets = np.zeros(moves_played)
            for k in range(moves_played):
                target_idx = min(k + n_step, moves_played - 1)
                value_targets[k] = mcts_values[target_idx] * (1 if (target_idx - k) % 2 == 0 else -1)
            if is_adjudicated_draw:
                result = f"Adjudicated draw (n-step={n_step}) in {moves_played} moves"
            else:
                result = f"Non-terminal (n-step={n_step}) in {moves_played} moves"
        else:
            value_targets = np.zeros(moves_played)
            result = f"Draw in {moves_played} moves"

        print(result)
        policy_targets = np.stack(policy_targets)
        return pos_list, policy_targets, value_targets
        