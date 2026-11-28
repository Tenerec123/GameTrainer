from .game import Game
from typing import Protocol
import math
import random

def dirichlet(alphas):
    gammas = [random.gammavariate(a, 1) for a in alphas]
    total = sum(gammas)
    return [g / total for g in gammas]

class Evaluator(Protocol):
    """Get the position and return a value for that position and the weights for each possible move.

    Value: 1 if the player that moves is winning and -1 if the other player is winning, 0 if the position is equal for both. 
    The intermediate values refer to the different scales of having a winning or a losing position, with the tie in the middle.
    Weights: Each move has a weight, all weights sum up to 1. The higher the weight, the better the move for the player that is moving.
    """
    def evaluate(self, position) -> tuple[dict, float]:
        """Returns a tuple with weights and value in that order"""

def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score

def root_noise(root_node, alpha=0.3, epsilon=0.25):
    if not root_node.is_expanded(): return
        
    children = list(root_node.children.values())
    n_actions = len(children)
    
    if n_actions <= 1: return
   
    noise = dirichlet([alpha] * n_actions)
    
    # Modificamos los priors en caliente
    for i, child in enumerate(children):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

class RolloutEvaluator:
    """Uniform priors + value from a random rollout to the end.

    No torch dependency: usable as a zero-training baseline evaluator and
    as the engine behind ``Player(rollout_eval=True)``.
    """

    def __init__(self, game):
        self.game = game

    def _legal(self, position):
        return list(self.game.get_legal_moves(position))

    def evaluate(self, position):
        legal = self._legal(position)
        if not legal:
            return {}, 0.0
        priors = {move: 1.0 / len(legal) for move in legal}
        pos = position
        while not self.game.is_game_over(pos):
            moves = self._legal(pos)
            if not moves:
                break
            pos = self.game.make_move(pos, random.choice(moves))
        value = self.game.get_result(pos, getattr(position, "turn", 1))
        return priors, float(value if value is not None else 0.0)


class Node:
    def __init__(self, prior, to_play):
        self.to_play = to_play
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count
    
    def select_child(self):
        max_ucb = -float('inf')
        best_move = None
        best_child = None
        for move, child in self.children.items():
            ucb = ucb_score(self, child)
            if ucb > max_ucb: 
                max_ucb = ucb
                best_move = move
                best_child = child
        return best_move, best_child
    def expand(self, state, move_probs:dict, child_turns=None):
        """Expand the node with the given move priors.

        child_turns: optional list aligned with move_probs (same order) giving
        the player to move in each child (forced continuations). When None,
        the turn always alternates. A dict move->turn is also accepted.
        """
        self.state = state
        moves = list(move_probs.keys())
        if child_turns is None:
            turns = [-self.to_play] * len(moves)
        elif isinstance(child_turns, dict):
            turns = [child_turns[m] for m in moves]
        else:
            turns = list(child_turns)
        for i, (move, prior) in enumerate(move_probs.items()):
            self.children[move] = Node(prior, turns[i])

    def is_expanded(self) -> bool:
        return len(self.children) > 0

class MCTS:
    def __init__(self, game:Game, evaluator:Evaluator, num_simulations):
        self.game = game
        self.evaluator = evaluator
        self.num_simulations = num_simulations

    def run(self, root:Node, state):
        # Terminal or no-move positions need no search.
        if self.game.is_game_over(state):
            value = self.game.get_result(state, getattr(root, "to_play", 1))
            if value is None:
                raise ValueError("game is malformed: the game is over but there is no terminal value")
            if root.visit_count == 0:
                root.visit_count = 1
                root.value_sum = value
            return None

        if not root.is_expanded():
            legal = self.game.get_legal_moves(state)
            n_legal = len(legal)
            if n_legal == 0:
                raise ValueError("game is malformed: no legal moves from a non-terminal position")
            if n_legal == 1:
                move = legal[0]
                move_probs = {move: 1.0}
                turns = self.game.next_turns(state, [move])
                root.expand(state, move_probs, turns)
            else:
                move_probs, value = self.evaluator.evaluate(state)
                moves = list(move_probs.keys())
                turns = self.game.next_turns(state, moves)
                root.expand(state, move_probs, turns)
            root.state = state
        if not root.children:
            raise ValueError("game is malformed: no legal moves from a non-terminal position")
        root_noise(root)
        for _ in range(self.num_simulations - root.visit_count):
            node = root
            search_path = [root]
            try:
                while node.is_expanded() and node.children:
                    move, child = node.select_child()
                    search_path.append(child)
                    node = child
                if len(search_path) < 2:
                    break
                node.state = self.game.make_move(search_path[-2].state, move)
                value = self.game.get_result(node.state, node.to_play)
                if value is None:
                    if len(self.game.get_legal_moves(node.state)) == 0:
                        raise ValueError("game is malformed: no legal moves from a non-terminal position")
                    move_probs, value = self.evaluator.evaluate(node.state)
                    moves = list(move_probs.keys())
                    turns = self.game.next_turns(node.state, moves)
                    node.expand(node.state, move_probs, turns)

                self.backpropagate(search_path, value, node.to_play)
            except Exception as e:
                print(e)
                # Malformed games must surface, not be swallowed.
                if isinstance(e, ValueError):
                    raise
                if len(search_path) > 1:
                    return [n.state for n in search_path[:-1]]
                return None
        return None
    def backpropagate(self, search_path, value, to_play):
        for node in search_path:
            node.value_sum += value * (-1)**(to_play != node.to_play)
            node.visit_count+=1
