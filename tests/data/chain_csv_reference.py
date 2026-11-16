"""Verbatim fixture: the four functions of the chain CSV worker that built the later submissions' CSVs (worker file sha256
ada24af501c219a3e8cc2a7db41f08d928601f3e2fad449ff67edd8f9a8edb07, the same file in every covered call from f1c15u on;
the earlier chain calls' worker has the same four functions). Tests compare the package's chain stage with them.
Not imported by the package."""
import numpy as np

DEN = 510


def test_sources(pins):
    """Archive names of the test maps in the per-image map dict: NEW{k}_{j} = fold-k model of map set j."""
    return [f'NEW{k}_{j}' for k in range(5) for j in range(int(pins['sets']))]


def chain_mf(maps, names):
    """The weight arms' MF probability map (float32) for one image: the package's mf_levels_mean over every set's fold-model maps."""
    from filseg.reproduction.arithmetic import mf_levels_mean
    return mf_levels_mean(*(maps[n] for n in names))


def weighted_blend(w, check_against=None):
    """blend_with_mf with the weights (w, (1 - w)/2, (1 - w)/2): the same checks, insertion order and Python sum. With
    check_against (the package function), every call asserts bit equality (used for w = 0.25)."""
    u = (1 - w) / 2

    def blend(full, local, mf):
        if full.dtype != np.uint8 or local.dtype != np.uint8: raise ValueError('U-Net maps must be uint8')
        if mf.dtype != np.float32 or not full.shape == local.shape == mf.shape: raise ValueError('Map shapes differ')
        p = sum(np.float32(x) * m for x, m in ((w, mf), (u, full.astype(np.float32) / 255), (u, local.astype(np.float32) / 255)))
        if check_against is not None: assert np.array_equal(p, check_against(full, local, mf)), 'W25_BLEND_DIFFERS'
        return p
    blend.weight = w
    return blend


class UnetMean:
    """chain2u: per-image uint8 mean of m U-Net test archives (m = 5 fold models x n sets): rint(level sum / m)."""
    def __init__(self, archives): self.archives = list(archives)

    def __getitem__(self, key):
        a = [z[key] for z in self.archives]
        if any(x.dtype != np.uint8 or x.shape != a[0].shape for x in a): raise ValueError('U-Net arrays must be matching uint8 maps')
        return np.rint(sum(x.astype(np.int64) for x in a) / len(a)).astype(np.uint8)
