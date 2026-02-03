from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import zarr
import torch

# ---- import BaseLowdimDataset (repo versions may differ) ----
BaseLowdimDataset = None
_import_errors = []

for _p in [
    "diffusion_policy.dataset.base_dataset",           # common
    "diffusion_policy.dataset.base_lowdim_dataset",    # alt
    "diffusion_policy.dataset.dataset_base",           # alt
]:
    try:
        mod = __import__(_p, fromlist=["BaseLowdimDataset"])
        BaseLowdimDataset = getattr(mod, "BaseLowdimDataset")
        break
    except Exception as e:
        _import_errors.append(( _p, repr(e) ))

if BaseLowdimDataset is None:
    raise ImportError(
        "Cannot import BaseLowdimDataset. Tried:\n"
        + "\n".join([f"  - {_p}: {err}" for _p, err in _import_errors])
        + "\nPlease locate where BaseLowdimDataset is defined in your diffusion_policy repo "
          "and update the import paths list accordingly."
    )

from diffusion_policy.model.common.normalizer import LinearNormalizer


def _load_zarr_arrays(zarr_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    obs = np.array(root["data"]["obs"], dtype=np.float32)          # (N, obs_dim)
    act = np.array(root["data"]["action"], dtype=np.float32)       # (N, act_dim)
    episode_ends = np.array(root["meta"]["episode_ends"], dtype=np.int64)
    assert obs.shape[0] == act.shape[0]
    return obs, act, episode_ends


def _build_episode_ranges(episode_ends: np.ndarray) -> np.ndarray:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return np.stack([starts, episode_ends], axis=1)  # (E,2) global [start,end)


@dataclass
class _IndexItem:
    ep_id: int
    t0: int


class PickPlaceLowdimDataset(BaseLowdimDataset):
    """
    Minimal BaseLowdimDataset implementation.
    Returns:
      sample["obs"]    : (n_obs_steps, obs_dim)
      sample["action"] : (horizon, act_dim)
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        train: bool = True,
        val_ratio: float = 0.1,
        seed: int = 0,
        # keep these for compatibility with existing configs
        pad_before: int = 0,
        pad_after: int = 0,
        # some repos pass these
        **kwargs,
    ):
        super().__init__()
        if not zarr_path or not isinstance(zarr_path, str):
            raise ValueError(f"zarr_path is empty/invalid: {zarr_path!r}")

        self.zarr_path = zarr_path
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.train = bool(train)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)

        obs_all, act_all, episode_ends = _load_zarr_arrays(zarr_path)
        self._obs_all = obs_all
        self._act_all = act_all
        self._ep_ranges = _build_episode_ranges(episode_ends)

        self.obs_dim = int(obs_all.shape[1])
        self.action_dim = int(act_all.shape[1])

        # split by episode
        E = self._ep_ranges.shape[0]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(E)
        n_val = max(1, int(round(E * self.val_ratio)))
        self._val_eps = set(perm[:n_val].tolist())

        # build indices
        self._index: List[_IndexItem] = []
        for ep_id, (s, e) in enumerate(self._ep_ranges):
            is_val = ep_id in self._val_eps
            if self.train and is_val:
                continue
            if (not self.train) and (not is_val):
                continue

            ep_len = int(e - s)
            # allow all t0; we will edge-pad inside episode
            for t0 in range(ep_len):
                self._index.append(_IndexItem(ep_id=ep_id, t0=t0))

        self._normalizer: Optional[LinearNormalizer] = None

    # --- required/expected by workspace ---
    def get_validation_dataset(self) -> "PickPlaceLowdimDataset":
        return PickPlaceLowdimDataset(
            zarr_path=self.zarr_path,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            train=False,
            val_ratio=self.val_ratio,
            seed=self.seed,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )

    def get_normalizer(self) -> LinearNormalizer:
        if self._normalizer is not None:
            return self._normalizer

        # fit on TRAIN split only
        mask = np.ones((self._obs_all.shape[0],), dtype=bool)
        for ep_id, (s, e) in enumerate(self._ep_ranges):
            if ep_id in self._val_eps:
                mask[s:e] = False

        obs = self._obs_all[mask]
        act = self._act_all[mask]

        n = LinearNormalizer()
        # most versions accept dict fit
        n.fit({"obs": obs, "action": act}, mode="limits")
        self._normalizer = n
        return self._normalizer

    # --- Dataset interface ---
    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _slice_pad(arr: np.ndarray, start: int, end: int) -> np.ndarray:
        # arr: (T,D) episode-local
        T = arr.shape[0]
        out_len = end - start
        s0 = max(0, start)
        e0 = min(T, end)
        chunk = arr[s0:e0]
        # pad left
        if start < 0:
            pad = -start
            first = chunk[:1] if len(chunk) else arr[:1]
            chunk = np.concatenate([np.repeat(first, pad, axis=0), chunk], axis=0)
        # pad right
        if end > T:
            pad = end - T
            last = chunk[-1:] if len(chunk) else arr[-1:]
            chunk = np.concatenate([chunk, np.repeat(last, pad, axis=0)], axis=0)
        assert chunk.shape[0] == out_len
        return chunk

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._index[idx]
        ep_s, ep_e = self._ep_ranges[item.ep_id]
        ep_obs = self._obs_all[ep_s:ep_e]
        ep_act = self._act_all[ep_s:ep_e]

        t0 = int(item.t0)

        # obs history ending at t0 (inclusive)
        obs_end = t0 + 1
        obs_start = obs_end - self.n_obs_steps

        # action horizon starting at t0
        act_start = t0
        act_end = t0 + self.horizon

        obs_seq = self._slice_pad(ep_obs, obs_start, obs_end)   # (n_obs_steps, obs_dim)
        act_seq = self._slice_pad(ep_act, act_start, act_end)   # (horizon, act_dim)

        return {
            "obs": torch.from_numpy(obs_seq).float(),
            "action": torch.from_numpy(act_seq).float(),
        }
