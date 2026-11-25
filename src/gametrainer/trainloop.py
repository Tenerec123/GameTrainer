"""Training helpers for gametrainer: dataset, masked loss, epoch loop, checkpoints.

Torch is imported lazily so that ``import gametrainer`` never requires torch.
Install the optional extra with ``pip install 'gametrainer[nn]'``.

This is an intermediate, "make it work" module — not production code. The full
AlphaZero-style pipeline (self-play collection + training + checkpointing) lives
in the training notebook; this module gives a Trainer class that wraps the
shared net, a dataset and the training step so a programmer can drive training
programmatically.
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    torch = None
    nn = None
    F = None
    Dataset = None
    DataLoader = None
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Trainer requires torch. Install with: pip install 'gametrainer[nn]'"
        )


class PolicyValueLoss(nn.Module if _TORCH_AVAILABLE else object):
    """Loss = value_weight * MSE(value_pred, value_target) + policy_weight * masked CE.

    The mask is applied inside the loss (policy logits are masked with -1e9
    before log_softmax), exactly like inference masks with -inf but stable
    under autograd.
    """

    def __init__(self, policy_weight=1.0, value_weight=1.0):
        _require_torch()
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(self, policy_logits, value_pred, policy_target, value_target, mask):
        _require_torch()
        batch = policy_logits.shape[0]

        # Value: MSE. reshape(-1) makes both (N,) so the subtraction is safe.
        value_loss = self.value_weight * (
            (value_pred.reshape(-1) - value_target.reshape(-1)) ** 2
        ).mean()

        # Policy: masked cross-entropy over the flattened policy space.
        flat_logits = policy_logits.reshape(batch, -1)
        flat_target = policy_target.reshape(batch, -1)
        flat_mask = mask.reshape(batch, -1)

        masked_logits = flat_logits.masked_fill(flat_mask == 0, -1e9)
        log_pred = F.log_softmax(masked_logits, dim=-1)
        policy_loss = -(flat_target * log_pred).sum(dim=-1).mean() * self.policy_weight

        return value_loss + policy_loss, policy_loss, value_loss


class GameDataset(Dataset if _TORCH_AVAILABLE else object):
    """Dataset that stores positions and re-encodes board input + mask on read."""

    def __init__(self, game, positions, policy_targets, value_targets):
        _require_torch()
        self.game = game
        self.positions = list(positions)
        self.policy_targets = (
            np.asarray(policy_targets, dtype=np.float32)
            if len(self.positions) > 0
            else np.zeros((0,), dtype=np.float32)
        )
        self.value_targets = (
            np.asarray(value_targets, dtype=np.float32)
            if len(self.positions) > 0
            else np.zeros((0,), dtype=np.float32)
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx]
        board = torch.from_numpy(
            np.asarray(self.game.get_board_input(position), dtype=np.float32)
        )
        mask = torch.from_numpy(
            np.asarray(self.game.get_move_mask(position), dtype=np.float32)
        )
        policy = torch.from_numpy(
            np.asarray(self.policy_targets[idx], dtype=np.float32)
        )
        value = torch.tensor(self.value_targets[idx], dtype=torch.float32)
        return board, mask, policy, value


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns (loss, policy_loss, value_loss) means."""
    _require_torch()
    model.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    n_batches = 0

    for boards, masks, policy_targets, value_targets in loader:
        boards = boards.to(device)
        masks = masks.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad()
        policy_logits, value = model(boards)
        loss, p_loss, v_loss = criterion(
            policy_logits, value, policy_targets, value_targets, masks
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_policy += float(p_loss.item())
        total_value += float(v_loss.item())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_batches, total_policy / n_batches, total_value / n_batches


class Trainer:
    """Intermediate trainer wrapping a shared net + dataset + training step.

    The net is the shared artifact: the same ``nn.Module`` is used by
    ``NeuralEvaluator`` (inference door) and by this Trainer (training door).
    Pass an existing net to share weights, or leave ``net=None`` to build a
    small generic net from ``game``.

    Dataset is the MatchMaker output: a tuple
    ``(pos_list, policy_targets, value_targets)`` (all numpy/objects), or
    ``None`` for an empty trainer.
    """

    def __init__(
        self,
        game,
        net=None,
        dataset=None,
        device=None,
        lr=3e-4,
        weight_decay=1e-4,
        policy_weight=1.0,
        value_weight=1.0,
        batch_size=256,
        max_dataset_size=100000,
    ):
        _require_torch()
        from .model import NeuralEvaluator, build_net

        self.game = game

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Ergonomics: allow Trainer(game, dataset) by detecting a non-module.
        if net is not None and not hasattr(net, "parameters") and isinstance(net, (tuple, list)):
            dataset = net
            net = None

        if net is None:
            net = build_net(game, "small")
        self.net = net.to(self.device)

        self.evaluator = NeuralEvaluator(game, net=self.net, device=self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = PolicyValueLoss(policy_weight, value_weight)
        self.batch_size = batch_size
        self.max_dataset_size = max_dataset_size

        # Dataset storage (in memory, numpy-friendly).
        self.positions = []
        self.policy_targets = None
        self.value_targets = None
        if dataset is not None:
            self.set_dataset(dataset)

    # ------------------------------------------------------------------ data
    def set_dataset(self, dataset):
        """Load a MatchMaker output: (pos_list, policy_targets, value_targets)."""
        if isinstance(dataset, (tuple, list)):
            positions, policy, values = dataset
        else:
            positions = getattr(dataset, "positions", [])
            policy = getattr(dataset, "policy_targets", None)
            values = getattr(dataset, "value_targets", None)

        self.positions = list(positions) if positions is not None else []
        if len(self.positions) > 0:
            self.policy_targets = np.asarray(policy, dtype=np.float32)
            self.value_targets = np.asarray(values, dtype=np.float32)
        else:
            self.policy_targets = None
            self.value_targets = None
        self._trim()

    def add_data(self, positions, policy_targets, value_targets):
        """Append positions + targets to the dataset (in memory)."""
        positions = list(positions) if positions is not None else []
        if len(positions) == 0:
            return
        policy = np.asarray(policy_targets, dtype=np.float32)
        values = np.asarray(value_targets, dtype=np.float32)

        if self.positions:
            self.positions.extend(positions)
            self.policy_targets = np.concatenate(
                [self.policy_targets, policy], axis=0
            )
            self.value_targets = np.concatenate(
                [self.value_targets, values], axis=0
            )
        else:
            self.positions = positions
            self.policy_targets = policy
            self.value_targets = values
        self._trim()

    def collect(self, matchmaker, n_moves, n_step=5):
        """Run self-play via ``matchmaker`` until ``n_moves`` positions gathered.

        Appends the collected data to the current dataset, calling ``add_data``
        once per finished match. The whole last match is kept even if it pushes
        the total past ``n_moves`` (no trimming of the surplus). ``n_step`` is
        forwarded to ``MatchMaker.run_match``.
        """
        added = 0
        empty_streak = 0
        while added < n_moves:
            pos_list, pol, val = matchmaker.run_match(n_step)
            if not pos_list:
                empty_streak += 1
                if empty_streak > 10:
                    raise RuntimeError(
                        "collect: matchmaker returned no data too many times "
                        "(terminal initial position?)"
                    )
                continue
            empty_streak = 0

            pos_list = list(pos_list)
            pol = np.asarray(pol, dtype=np.float32)
            val = np.asarray(val, dtype=np.float32)

            self.add_data(pos_list, pol, val)
            added += len(pos_list)

    def _trim(self):
        if self.positions and len(self.positions) > self.max_dataset_size:
            keep = self.max_dataset_size
            self.positions = self.positions[-keep:]
            self.policy_targets = self.policy_targets[-keep:]
            self.value_targets = self.value_targets[-keep:]

    def __len__(self):
        return len(self.positions)

    # --------------------------------------------------------------- training
    def make_loader(self, batch_size=None, shuffle=True):
        if len(self.positions) == 0:
            raise ValueError("dataset is empty")
        bs = batch_size or self.batch_size
        dataset = GameDataset(
            self.game, self.positions, self.policy_targets, self.value_targets
        )
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle)

    def train(self, epochs=1, batch_size=None, shuffle=True, verbose=True):
        """Train for ``epochs`` epochs over the current dataset.

        Returns a list of (loss, policy_loss, value_loss) per epoch.
        """
        if len(self.positions) == 0:
            raise ValueError("dataset is empty; collect or set data first")

        loader = self.make_loader(batch_size=batch_size, shuffle=shuffle)
        self.net.train()
        history = []
        for epoch in range(epochs):
            loss, p_loss, v_loss = train_epoch(
                self.net, loader, self.optimizer, self.criterion, self.device
            )
            history.append((loss, p_loss, v_loss))
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {loss:.4f} | Policy: {p_loss:.4f} | "
                    f"Value: {v_loss:.4f}"
                )
        self.net.eval()
        return history

    # -------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path="checkpoint.pt", save_dataset=False):
        """Save net + optimizer + step to ``path``. Returns path.

        If ``save_dataset=True``, the current in-memory dataset (positions +
        targets) is embedded so a later ``load_checkpoint(..., restore_dataset=True)``
        can resume training without re-collecting data.
        """
        checkpoint = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": len(self.positions),
        }
        if save_dataset:
            checkpoint["dataset"] = {
                "positions": list(self.positions),
                "policy": (
                    None
                    if self.policy_targets is None
                    else np.asarray(self.policy_targets)
                ),
                "values": (
                    None
                    if self.value_targets is None
                    else np.asarray(self.value_targets)
                ),
            }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path="checkpoint.pt", restore_dataset=False):
        """Load net + optimizer from ``path``. Returns True if loaded.

        If ``restore_dataset=True`` and the checkpoint contains a dataset, it is
        restored into this Trainer (positions + targets).

        Note: checkpoints may embed pickled game positions, so loading uses
        ``torch.load(..., weights_only=False)``. Only load checkpoints from
        trusted sources.
        """
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if restore_dataset and "dataset" in checkpoint:
            d = checkpoint["dataset"]
            positions = d.get("positions") or []
            policy = d.get("policy")
            values = d.get("values")
            if positions:
                self.set_dataset((positions, policy, values))
        return True

    # ----------------------------------------------------------- full pipeline
    def fulltrain(
        self,
        loops=1,
        new_moves=1000,
        max_moves=300,
        mcts_simulations=50,
        train_epochs=1,
        batch_size=None,
        checkpoint_path=None,
        resume=False,
        n_step=5,
        verbose=True,
    ):
        """All-in-one loop: collect self-play data, then train, then checkpoint.

        Reuses ``collect``, ``train`` and ``save_checkpoint`` instead of
        reimplementing them. Generation and training run sequentially per loop.
        Returns ``(positions, policy_targets, value_targets)``.
        """
        from .selfplay import MatchMaker

        bs = batch_size or self.batch_size

        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path, restore_dataset=True)
            if verbose:
                print(f"resumed from {checkpoint_path} ({len(self)} positions)")

        mm = MatchMaker(self.game, self.evaluator, max_moves, mcts_simulations)

        try:
            for loop in range(loops):
                if verbose:
                    print(f"--- loop {loop + 1}/{loops} ---")

                self.collect(mm, new_moves, n_step=n_step)
                if verbose:
                    print(f"dataset size: {len(self)}")

                self.train(epochs=train_epochs, batch_size=bs, verbose=verbose)

                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path, save_dataset=True)
                    if verbose:
                        print(f"checkpoint saved: {checkpoint_path}")

                if verbose:
                    print(f"loop {loop + 1} finished")

        except KeyboardInterrupt:
            if verbose:
                print("\n[!] stopped manually, saving checkpoint...")
            if checkpoint_path:
                try:
                    self.save_checkpoint(checkpoint_path, save_dataset=True)
                    if verbose:
                        print("[+] checkpoint saved")
                except Exception as e_save:
                    if verbose:
                        print(f"[-] checkpoint save failed: {e_save}")
        except Exception as e:
            if verbose:
                print(f"[-] unexpected error: {e}")
            if checkpoint_path:
                try:
                    self.save_checkpoint(checkpoint_path, save_dataset=True)
                    if verbose:
                        print("[+] emergency checkpoint saved")
                except Exception as e_save:
                    if verbose:
                        print(f"[-] emergency save failed: {e_save}")
            raise

        self.net.eval()
        policy = (
            self.policy_targets
            if self.policy_targets is not None
            else np.zeros((0,), dtype=np.float32)
        )
        values = (
            self.value_targets
            if self.value_targets is not None
            else np.zeros((0,), dtype=np.float32)
        )
        return self.positions, policy, values


__all__ = [
    "PolicyValueLoss",
    "GameDataset",
    "train_epoch",
    "Trainer",
]
