"""Probability-map sources: one model's saved maps, or an average of several.

Training kernels persist sigmoid maps as uint8 npz archives (`p * 255`), at
1024 px for the whole-frame models and 2048 px for the tiled ones. Both feed
the same decision rule, because `instancing.upsample` passes a full-resolution
map through and bilinearly upsamples a half-resolution one.

`EnsembleMaps` is that fact used deliberately: members are brought to the 2048
px grid *before* averaging — the grid the rule consumes anyway — so a
mixed-resolution ensemble is scored by the identical loop as a single model
(ADR-0009 Arm A), with no separate code path to drift.

Maps are read one file at a time (npz members are lazily decompressed), so an
N-member ensemble over 707 files holds N full-frame maps, not 707N.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator, Sequence

import numpy as np

from .instancing import upsample


class Uint8Maps(Mapping):
    """A single kernel's `oof_prob.npz`, dequantized to [0, 1] float32."""

    def __init__(self, npz, keys: Sequence[str] | None = None):
        self._z = npz
        if keys is None:
            keys = npz.files if hasattr(npz, "files") else list(npz)
        self._keys = sorted(keys)

    def __getitem__(self, f: str) -> np.ndarray:
        return self._z[f].astype(np.float32) / 255.0

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class EnsembleMaps(Mapping):
    """Weighted mean of several members' maps, on the full 2048 px grid.

    Members may differ in resolution; each is upsampled first, so the average
    is taken on the grid the decision rule consumes rather than on whichever
    grid a member happened to be trained at.
    """

    def __init__(self, members: Sequence, weights: Sequence[float] | None = None):
        assert members, "an ensemble needs at least one member"
        self._m = list(members)
        w = list(weights) if weights is not None else [1.0] * len(self._m)
        assert len(w) == len(self._m), "one weight per member"
        assert all(x > 0 for x in w), "weights must be positive"
        self._w = [x / sum(w) for x in w]
        keys = [set(m.files if hasattr(m, "files") else m) for m in self._m]
        common = set.intersection(*keys)
        assert common, "members share no files"
        for i, k in enumerate(keys):
            assert k == common, f"member {i} covers a different file set"
        self._keys = sorted(common)

    def __getitem__(self, f: str) -> np.ndarray:
        acc = None
        for m, w in zip(self._m, self._w):
            p = m[f]
            if p.dtype == np.uint8:
                p = p.astype(np.float32) / 255.0
            p = upsample(p)
            acc = p * w if acc is None else acc + p * w
        return acc

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
