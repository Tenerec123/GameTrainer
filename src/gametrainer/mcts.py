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
    def expand(self, state, move_probs:dict):
        self.state = state
        for i, (move, prior) in enumerate(move_probs.items()):
            self.children[move] = Node(prior, -self.to_play)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

class MCTS:
    def __init__(self, game:Game, evaluator:Evaluator, num_simulations):
        self.game = game
        self.evaluator = evaluator
        self.num_simulations = num_simulations

    def run(self, root:Node, state):
        if not root.is_expanded():
            move_probs, _ = self.evaluator.evaluate(state)
            root.expand(state, move_probs)
        root_noise(root)
        for _ in range(self.num_simulations - root.visit_count):
            node = root
            search_path = [root]
            try:
                while node.is_expanded():
                    move,child = node.select_child()
                    search_path.append(child)
                    node = child
                node_state = self.game.make_move(search_path[-2].state, move)
                value = self.game.get_result(node_state, node.to_play)
                if value is None:
                    move_probs, value = self.evaluator.evaluate(node_state)
                    node.expand(node_state, move_probs)
                self.backpropagate(search_path, value, node.to_play)
            except ValueError as e:
                print(e)
                return [node.state for node in search_path[:-1]]
        return None
    def backpropagate(self, search_path, value, to_play):
        for node in search_path:
            node.value_sum += value * (-1)**(to_play != node.to_play)
            node.visit_count+=1
