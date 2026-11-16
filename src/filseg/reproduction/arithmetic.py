"""Native quantization and ordered float32 candidate blend."""
import numpy as np


def mf_mean(a,b,c):
    if not (a.dtype==b.dtype==c.dtype==np.uint16 and a.shape==b.shape==c.shape):
        raise ValueError('MF arrays must be matching uint16 maps')
    if any(x.max()>510 for x in (a,b,c)):raise ValueError('MF quantization exceeds 510')
    return (a+b+c).astype(np.float32)/np.float32(1530)


def candidate_blend(full,local,mf0,mf1,mf2):
    if full.dtype!=np.uint8 or local.dtype!=np.uint8:raise ValueError('U-Net maps must be uint8')
    if not full.shape==local.shape==mf0.shape:raise ValueError('Map shapes differ')
    # This insertion order and Python sum reproduce the shipping worker.
    return sum(np.float32(w)*m for w,m in (
        (.25,mf_mean(mf0,mf1,mf2)),(.375,full.astype(np.float32)/255),(.375,local.astype(np.float32)/255)))


def mf_levels_mean(*maps):
    """MF probability of k uint16 maps (each value / 510): the level sum / (510 k), float32. Of three fold maps it is mf_mean bit for bit."""
    if not maps or any(m.dtype!=np.uint16 or m.shape!=maps[0].shape for m in maps):raise ValueError('MF arrays must be matching uint16 maps')
    if any(m.max()>510 for m in maps):raise ValueError('MF quantization exceeds 510')
    # Integer sum (at most 6 x 510 levels fits uint16), then one float32 division as in mf_mean.
    return sum(maps[1:],maps[0].copy()).astype(np.float32)/np.float32(510*len(maps))


def blend_with_mf(full,local,mf):
    """candidate_blend with the MF term given as a float32 probability map (same insertion order and Python sum)."""
    if full.dtype!=np.uint8 or local.dtype!=np.uint8:raise ValueError('U-Net maps must be uint8')
    if mf.dtype!=np.float32 or not full.shape==local.shape==mf.shape:raise ValueError('Map shapes differ')
    return sum(np.float32(w)*m for w,m in (
        (.25,mf),(.375,full.astype(np.float32)/255),(.375,local.astype(np.float32)/255)))
