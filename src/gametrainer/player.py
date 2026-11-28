"""Post-training ``Player`` (agent) shell for gametrainer.

A ``Player`` wraps a trained network behind the ``NeuralEvaluator`` inference
door and (optionally) an MCTS search, exposing a small, gameplay-oriented API:
``evaluate``, ``value`` and ``best_move``. Moves are already decoded (no
matrices) — the Player operates in real move space, exactly like
``NeuralEvaluator``.

Torch is imported lazily so that ``import gametrainer`` never requires torch.
"""

import os

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    torch = None
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Player requires torch. Install with: pip install 'gametrainer[nn]'"
        )


class Player:
    """Post-training shell: decode moves, evaluate positions, pick moves.

    Parameters
    ----------
    game:
        The game (must be an ``EncodableGame`` when using the NN path).
    evaluator:
        Optional pre-built evaluator. If given, ``net``/``checkpoint``/``arch``
        are ignored.
    net:
        Optional torch module to wrap. Ignored if ``evaluator`` is given.
    checkpoint / checkpoint_path / path / weights / weights_path / ckpt:
        Optional path to a checkpoint to load into ``net``. Accepts Trainer
        checkpoints (with a ``"model"`` key) or a raw state_dict.
    arch:
        Preset name passed to ``build_net`` when no ``net`` is given.
    device:
        ``"cpu"`` / ``"cuda"`` or a torch device.
    mcts_simulations / simulations / num_simulations / mcts_sims / sims:
        If > 0, build an MCTS search with that many simulations and use it in
        ``best_move``. Otherwise ``best_move`` is the argmax of the priors.
    """

    _SIM_KEYS = ("mcts_simulations", "simulations", "num_simulations",
                 "mcts_sims", "sims")
    _CKPT_KEYS = ("checkpoint", "checkpoint_path", "path", "weights",
                  "weights_path", "ckpt")
    _ROLLOUT_KEYS = ("rollout_eval", "rollout")

    def __init__(self, game, *args, **kwargs):
        # ---- pull flexible kwargs -----------------------------------------
        evaluator = kwargs.pop("evaluator", None)
        net = kwargs.pop("net", None)
        arch = kwargs.pop("arch", None)
        device = kwargs.pop("device", None)

        rollout_eval = False
        for k in self._ROLLOUT_KEYS:
            if k in kwargs:
                rollout_eval = bool(kwargs.pop(k))
                break

        mcts_simulations = 0
        mcts_explicit = False
        for k in self._SIM_KEYS:
            if k in kwargs:
                mcts_simulations = kwargs.pop(k)
                mcts_explicit = True
                break

        checkpoint = None
        for k in self._CKPT_KEYS:
            if k in kwargs:
                checkpoint = kwargs.pop(k)
                break

        # positional shorthand: Player(game, net) or Player(game, checkpoint)
        # or Player(game, simulations)
        if args:
            first = args[0]
            if isinstance(first, bool):
                rollout_eval = bool(first)
                args = args[1:]
            elif hasattr(first, "parameters"):
                net = first
                args = args[1:]
            elif isinstance(first, (str, os.PathLike)):
                checkpoint = first
                args = args[1:]
            if args:
                try:
                    mcts_simulations = int(args[0])
                    mcts_explicit = True
                    args = args[1:]
                except (TypeError, ValueError):
                    pass

        self.game = game

        if rollout_eval:
            # ignore net/checkpoint/arch/device: pure rollout evaluator
            from .mcts import RolloutEvaluator

            self.net = None
            self.evaluator = RolloutEvaluator(game)
            self.device = None
            if not mcts_explicit:
                mcts_simulations = 50
        elif evaluator is not None:
            self.evaluator = evaluator
            self.net = getattr(evaluator, "net", None)
            self.device = getattr(self.evaluator, "device", None)
        else:
            _require_torch()
            from .model import NeuralEvaluator, build_net

            if net is None:
                net = build_net(game, arch or "small")
            if device is None:
                device = "cpu"
            self.net = net
            self.evaluator = NeuralEvaluator(game, net=self.net, arch=None,
                                            device=device)
            self.device = getattr(self.evaluator, "device", None)
            if self.device is None:
                try:
                    self.device = next(self.net.parameters()).device
                except Exception:
                    self.device = None
            if checkpoint is not None:
                self.load_checkpoint(checkpoint)

        self.mcts_simulations = int(mcts_simulations) if mcts_simulations else 0

        if not hasattr(self, "device") or self.device is None:
            self.device = getattr(self.evaluator, "device", None)
            if self.device is None:
                try:
                    self.device = next(self.net.parameters()).device
                except Exception:
                    self.device = None

        # MCTS (optional)
        self.mcts = None
        if self.mcts_simulations and self.mcts_simulations > 0:
            from .mcts import MCTS

            self.mcts = MCTS(self.game, self.evaluator, self.mcts_simulations)

    # ------------------------------------------------------------------ io
    def load_checkpoint(self, path):
        """Load network weights from ``path``. Returns True on success."""
        _require_torch()
        device = getattr(self, "device", None)
        if device is None:
            device = getattr(getattr(self, "evaluator", None), "device", None)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.net.load_state_dict(state)
        self.net.eval()
        if self.device is not None:
            self.net.to(self.device)
        return True

    # --------------------------------------------------------------- inference
    def evaluate(self, position):
        """Return (priors dict, value float) for ``position``."""
        return self.evaluator.evaluate(position)

    def value(self, position):
        """Return the value estimate for ``position``."""
        _, v = self.evaluator.evaluate(position)
        return v

    # --------------------------------------------------------------- play
    def best_move(self, position, temperature=0.0, temp=None):
        """Pick a move.

        Uses MCTS when ``mcts_simulations > 0``, otherwise the argmax of the
        priors. ``temperature`` (``temp`` alias) controls exploration:
        0 (default) is greedy.
        """
        if temp is not None:
            temperature = temp

        if self.mcts is not None:
            from .mcts import Node

            root = Node(0, getattr(position, "turn", 1))
            self.mcts.run(root, position)
            if not root.children:
                moves = self.game.get_legal_moves(position)
                return moves[0] if isinstance(moves, list) and moves else None
            if temperature and temperature > 0:
                import random

                children = list(root.children.items())
                visits = [c.visit_count for _, c in children]
                if sum(visits) == 0:
                    visits = [1] * len(visits)
                weights = [v ** (1.0 / temperature) for v in visits]
                total = sum(weights)
                probs = [w / total for w in weights]
                idx = random.choices(range(len(children)), weights=probs, k=1)[0]
                return children[idx][0]
            return max(root.children.items(),
                       key=lambda kv: kv[1].visit_count)[0]

        priors, _ = self.evaluator.evaluate(position)
        if not priors:
            moves = self.game.get_legal_moves(position)
            return moves[0] if isinstance(moves, list) and moves else None
        if temperature and temperature > 0:
            import random

            items = list(priors.items())
            moves = [m for m, _ in items]
            weights = [max(p, 0.0) ** (1.0 / temperature) for _, p in items]
            if sum(weights) == 0:
                weights = [1.0] * len(weights)
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(moves, weights=probs, k=1)[0]
        return max(priors.items(), key=lambda kv: kv[1])[0]