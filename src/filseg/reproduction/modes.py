"""Named outputs of the submission stage: mode -> run_csv arguments -> the CSV sha256 it rebuilds from the deterministic maps.

Every CSV was built on Kaggle CPU from the map set d7aa6a41fafc26b07ce89b306b884f6667ebb68b86289ea4745ee44c0b0d195f and
submitted once (ref). All modes share the maps, the MF averaging, the blend and SHIP_R3; they differ only in which
candidates are kept (the selector) and how each kept mask is repaired. The MF-map modes below also change the MF map.

DEFAULT_MODE is the CSV the notebooks build when MODE is not set: our final submission, the chain mode 'ms125' (it reads
the stored maps of the release files). It is defined here only, so switching it (for example to 'pcombo', which needs only
the competition data) is this one line; any runnable mode works (tests/test_modes.py).
"""
DEFAULT_MODE = 'ms125'

MODES = {
    'pcombo': dict(label='P-combo selector (68 features, cutoff 0.35) + P-cand repair', ref=58278257,
                   sha256='35abbeed91108e40ddbd18c094c901d116441e363b10c31aa8092988d8e8f6e6', kwargs=dict(pnat=False, pcand=True, selector='pcombo')),
    'mfrich': dict(label='MF-rich selector (62 features, cutoff 0.375) + P-cand repair', ref=58274749,
                   sha256='5711532ebb1360bdbc27a2bb04fd1f388b4a6494d0ab911067775aa10c018b85', kwargs=dict(pnat=False, pcand=True, selector='mfrich')),
    'pneigh': dict(label='P-neigh selector (60 features, cutoff 0.375) + P-cand repair', ref=58268408,
                   sha256='c3edf2c8b681d42649dc7ccb6ed10e0400031b37c057d7d2798d6196bed0612c', kwargs=dict(pnat=False, pcand=True, selector='pneigh')),
    'fresh_ctrl': dict(label='Refit selector (54 features, cutoff 0.375) + P-cand repair', ref=58265972,
                       sha256='b01866b3f849d7bc43534c188b91545c26f8cd839493175d00ff6d25fb19d085', kwargs=dict(pnat=False, pcand=True, selector='fresh_ctrl')),
    'pcand': dict(label='LR39 selector (39 features, cutoff 0.35) + P-cand repair', ref=58232010,
                  sha256='14a5c1b684813081ad129d05e72fbb0c09592f292aeb1b1d84e121100c668c6b', kwargs=dict(pnat=False, pcand=True)),
    'pnat': dict(label='LR39 selector + P-nat repair (one threshold for every mask)', ref=58224552,
                 sha256='d54257e7f8a1f815d442037050233e52859b0f9594f128b7b258f368f4d05fbe', kwargs=dict(pnat=True, pcand=None)),
    'equivalent': dict(label='LR39 selector, no repair (incumbent-equivalent)', ref=58224327,
                       sha256='0b2123de5b614561ddf22c2bbc0f05000164ac6a595a9806104c7358b9f46b70', kwargs=dict(pnat=False, pcand=None)),
}

# MF-map variants (mfmaps.py): the P-combo recipe (68 features, StandardScaler + LR, P-cand repair) on a different MF map,
# read from the variant's map set (full and local of d7aa6a41... plus the variant's MF archives). The selector of each is
# pinned under the mode's name once fitted; until then a mode stops at refit.deployed. mftta: selector pinned, CSV hash of the
# Kaggle CPU ship kernel (mfmap_ship), submitted once (ref). mfhires and mfcomb were not adopted: no selector, no CSV hash.
from .mfmaps import EXPECTED_MAP_SETS as _MAP_SETS

MFMAP_MODES = {
    'mftta': dict(label='MF-TTA map (8-view D4 mean of each MF fold model) + P-combo selector + P-cand repair', ref=58299400,
                  sha256='3cdd3f06feec44b7b091d1807626c85dfbdcaf44afb285392e440b80432549ac',
                  map_set=_MAP_SETS['mftta'], kwargs=dict(pnat=False, pcand=True, selector='mftta', mf_variant='mftta')),
    'mfhires': dict(label='MF-hires map ((package MF map + tiled native-scale MF map) / 2) + P-combo selector + P-cand repair', ref=None, sha256=None,
                    map_set=_MAP_SETS['mfhires'], kwargs=dict(pnat=False, pcand=True, selector='mfhires', mf_variant='mfhires')),
    'mfcomb': dict(label='MF-comb map ((TTA map + tiled native-scale MF map) / 2) + P-combo selector + P-cand repair', ref=None, sha256=None,
                   map_set=_MAP_SETS['mfcomb'], kwargs=dict(pnat=False, pcand=True, selector='mfcomb', mf_variant='mfcomb')),
    # MF-var: the MF-TTA map and candidates, the P-combo 68 features plus 5 view-disagreement features of the 8 D4 views of
    # each MF fold model (averaged over MF0-MF2; MF{k}_views.npz of `infer-mfviews --views tta views`), cutoff 0.3, P-cand
    # repair. Selector and CSV hash are pinned once fitted and built; until its CSV is pinned it is not runnable().
    'mfvar': dict(label='MF-var: MF-TTA map + P-combo features + 5 view-disagreement features of the 8 D4 views (cutoff 0.3) + P-cand repair', ref=None, sha256=None,
                  map_set=_MAP_SETS['mftta'], kwargs=dict(pnat=False, pcand=True, selector='mfvar', mf_variant='mftta', mf_views=True)),
}


# Chain modes (chain.py): our later submissions, each a recipe (data/chain/<name>.json) of stored Mask2Former test-map
# archives, U-Net test maps, blend weight, selector and regressor; built from those stored maps (the release files) on CPU.
def _chain_modes():
    from .chain import load_recipe, names
    return {n: dict(label=r['label'], ref=r['submission']['ref'], sha256=r['csv']['sha256'], kwargs=dict(recipe=n))
            for n, r in ((n, load_recipe(n)) for n in names()) if r['csv']['sha256']}   # a recipe whose CSV is not pinned yet is not a mode


CHAIN_MODES = _chain_modes()


def runnable():
    """Every mode with a pinned CSV sha256, in documentation order: the 7 MODES, then the MF-map modes whose selector and
    CSV are pinned (mftta), then the chain modes. mfhires and mfcomb are not runnable."""
    return {**MODES, **{name: row for name, row in MFMAP_MODES.items() if row['sha256']}, **CHAIN_MODES}


def resolve(mode=None):
    """(name, row) of a runnable mode; None means DEFAULT_MODE."""
    name = DEFAULT_MODE if mode is None else mode
    table = runnable()
    if name not in table:raise ValueError('MODE must be one of ' + ', '.join(table) + '; got ' + repr(mode))
    return name, table[name]
