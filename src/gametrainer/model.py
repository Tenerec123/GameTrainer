"""Neural evaluator + generic conv network for gametrainer.

Torch is imported lazily so that ``import gametrainer`` (and the MCTS engine)
never requires torch. Install the optional extra with::

    pip install 'gametrainer[nn]'
"""

import warnings

import numpy as np
from .game import EncodableGame
try:
    import torch
    import torch.nn as nn 
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "NeuralEvaluator requires torch. Install with: pip install 'gametrainer[nn]'"
        )


def _probe_shapes(game: EncodableGame):
    pos0 = game.get_initial_position()
    board = np.asarray(game.get_board_input(pos0))
    mask = np.asarray(game.get_move_mask(pos0))

    if game.input_shape is None:
        raise ValueError("game must declare input_shape")
    if game.policy_shape is None:
        raise ValueError("game must declare policy_shape")

    if tuple(board.shape) != tuple(game.input_shape):
        raise ValueError(
            f"get_board_input shape {board.shape} does not match "
            f"declared input_shape {game.input_shape}"
        )
    if tuple(mask.shape) != tuple(game.policy_shape):
        raise ValueError(
            f"get_move_mask shape {mask.shape} does not match "
            f"declared policy_shape {game.policy_shape}"
        )
    return tuple(int(x) for x in game.input_shape), tuple(int(x) for x in game.policy_shape)


_PRESETS = {
    "tiny": (2, 32),
    "small": (4, 64),
    "medium": (6, 128),
    "large": (10, 256),
}


class _ResidualBlock(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, channels):
        _require_torch()
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class _GenericNet(nn.Module if _TORCH_AVAILABLE else object):
    """Shape-preserving residual tower + policy/value heads.

    Everything is shape-preserving (no pooling/strides), so the policy head
    emits exactly (C_pol, H, W) for any game. The net only knows shapes.
    """

    def __init__(self, c_in, c_pol, h, w, blocks, channels):
        _require_torch()
        super().__init__()
        self.stem = nn.Conv2d(c_in, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.tower = nn.ModuleList([_ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Conv2d(channels, c_pol, 1)
        v_channels = max(1, channels // 2)
        self.value_conv = nn.Conv2d(channels, v_channels, 1)
        self.value_fc1 = nn.Linear(v_channels * h * w, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        for block in self.tower:
            x = block(x)
        policy_logits = self.policy_head(x)  # (N, C_pol, H, W)
        v = F.relu(self.value_conv(x))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (N, 1)
        return policy_logits, v.squeeze(-1)


def build_net(game: EncodableGame, arch="small"):
    """Build a generic conv net for ``game`` using a named power preset."""
    _require_torch()
    input_shape, policy_shape = _probe_shapes(game)
    c_in, h, w = input_shape
    c_pol, ph, pw = policy_shape
    if (h, w) != (ph, pw):
        raise ValueError(
            f"input and policy spatial dims must match: {(h, w)} vs {(ph, pw)}"
        )
    if arch not in _PRESETS:
        raise ValueError(f"unknown arch {arch!r}; choose {list(_PRESETS)}")
    blocks, channels = _PRESETS[arch]
    return _GenericNet(c_in, c_pol, h, w, blocks, channels)


class NeuralEvaluator:
    """Wrap a (generic or user-supplied) net behind the Evaluator protocol.

    Inference door: takes positions, returns (priors dict, value float).
    The net itself is the shared artifact the Trainer also uses.
    The game must implement the EncodableGame contract (canonical space).
    """

    def __init__(self, game: EncodableGame, arch=None, net=None, device=None):
        _require_torch()
        self.game = game

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if net is not None:
            if arch is not None:
                warnings.warn(
                    f"arch={arch!r} is ignored because a custom net was provided",
                    stacklevel=2,
                )
            self.net = net
        elif arch is not None:
            self.net = build_net(game, arch)
        else:
            self.net = build_net(game)
        self.net.to(self.device)
        self.net.eval()

        if net is not None:
            self._validate_net()

    def _validate_net(self):
        """Fail fast if a custom net breaks the (input, logits, value) contract."""
        input_shape, policy_shape = _probe_shapes(self.game)
        x = torch.zeros(1, *input_shape, device=self.device)
        try:
            with torch.no_grad():
                logits, value = self.net(x)
        except (RuntimeError, TypeError, ValueError) as e:
            raise ValueError(
                f"custom net failed the dummy forward with input {tuple(x.shape)}: {e}"
            ) from e
        if tuple(logits.shape) != (1, *policy_shape):
            raise ValueError(
                f"net returned logits {tuple(logits.shape)}, expected {(1, *policy_shape)}"
            )
        if tuple(value.shape) not in ((1,), (1, 1)):
            raise ValueError(
                f"net returned value {tuple(value.shape)}, expected (1,) or (1, 1)"
            )

    def evaluate(self, position):
        _require_torch()
        board = np.asarray(self.game.get_board_input(position), dtype=np.float32)
        mask = np.asarray(self.game.get_move_mask(position), dtype=np.float32)

        x = torch.from_numpy(board).unsqueeze(0).to(self.device)  # (1, C, H, W)
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)  # (1, C_pol, H, W)

        with torch.no_grad():
            logits, value = self.net(x)

        logits_flat = logits.reshape(1, -1)
        mask_flat = m.reshape(1, -1)

        if float(mask_flat.sum().item()) == 0.0:
            policy = np.zeros(logits.shape[1:], dtype=np.float32)
        else:
            neg_inf = torch.full_like(logits_flat, float("-inf"))
            masked = torch.where(mask_flat > 0, logits_flat, neg_inf)
            probs_flat = torch.softmax(masked, dim=-1)
            policy = probs_flat.reshape(logits.shape[1:]).detach().cpu().numpy().astype(np.float32)

        to_play = getattr(position, "turn", 1)
        priors = self.game.get_move_probs(policy, to_play)
        v = float(value.item()) if hasattr(value, "item") else float(value)
        return priors, v
