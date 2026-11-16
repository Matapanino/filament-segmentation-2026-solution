"""GC1: activation checkpointing / channels_last on the real HF Mask2Former (Swin-T, pretrained), CPU, tiny crop.

Needs transformers 4.57.1 + the cached pretrained weights (offline: HF_HUB_OFFLINE=1). Skipped where transformers is absent
(e.g. a lightweight test environment). Checks: (1) the default cfg applies nothing (loss and grads bit-identical); (2) checkpointing each part
gives the same loss and parameter gradients (fp32 and bf16 autocast) and really drops saved activations;
(3) channels_last gives the same loss/grads within fp32 conv rounding.
"""
import copy

import pytest
import torch

pytest.importorskip('transformers')
import model as M  # noqa: E402
import train  # noqa: E402

S = 160  # crop side for the CPU check


@pytest.fixture(scope='module')
def base():
    torch.manual_seed(0)
    return M.build_pretrained('swin-tiny')


def inputs():
    g = torch.Generator().manual_seed(5)
    x = torch.rand(2, 1, S, S, generator=g)
    ys = []
    for n in (3, 2):
        m = torch.zeros(n, S, S)
        for i in range(n):
            r0, c0 = (int(v) for v in torch.randint(0, S - 40, (2,), generator=g))
            m[i, r0:r0 + 20 + 5 * i, c0:c0 + 30] = 1
        ys.append(m)
    return x, ys


def run(d, autocast=False, tally=False):
    d.train()
    d.zero_grad(set_to_none=True)
    x, ys = inputs()
    saved = [0]
    params = {p.data_ptr() for p in d.parameters()}

    def pack(t):
        if t.data_ptr() not in params:
            saved[0] += t.numel() * t.element_size()
        return t
    torch.manual_seed(1)  # drop_path + point sampling draw from the global RNG
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t) if tally else _null():
        with torch.autocast('cpu', dtype=torch.bfloat16) if autocast else _null():
            loss = d(x, ys).loss
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in d.named_parameters() if p.grad is not None}
    return loss.detach(), grads, saved[0]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def max_rel(ga, gb):
    """Relative L2 difference of all parameter gradients together, ||a - b|| / ||a||."""
    assert ga.keys() == gb.keys()
    num = sum(float((ga[k].double() - gb[k].double()).pow(2).sum()) for k in ga)
    den = sum(float(ga[k].double().pow(2).sum()) for k in ga)
    return (num / den) ** 0.5


def test_default_cfg_applies_nothing(base):
    ref = run(copy.deepcopy(base))
    d = copy.deepcopy(base)
    cfg = train.config_from_args(train.build_parser().parse_args(['--run-dir', 'r', '--data-root', 'd',
                                                                  '--fold-table', 'f']))
    assert M.apply_opt_in(d, cfg) == {}
    assert not any(getattr(m, 'gradient_checkpointing', False) for m in d.modules())
    assert not any('forward' in vars(m) for m in d.modules())
    got = run(d)
    assert torch.equal(ref[0], got[0])
    assert all(torch.equal(ref[1][k], got[1][k]) for k in ref[1])


@pytest.mark.parametrize('spec', ['backbone', 'pixel', 'decoder', 'all'])
def test_ckpt_same_loss_and_grads_fp32(base, spec):
    ref = run(copy.deepcopy(base), tally=True)
    d = copy.deepcopy(base)
    receipt = M.apply_opt_in(d, dict(grad_ckpt=spec))
    assert receipt['grad_ckpt'] and all(v > 0 for v in receipt['grad_ckpt'].values())
    got = run(d, tally=True)
    print(f'GC1 fp32 {spec}: loss {float(ref[0]):.9g} vs {float(got[0]):.9g}, rel L2 grad diff '
          f'{max_rel(ref[1], got[1]):.3g}, saved activations {ref[2]} -> {got[2]} B')
    assert torch.allclose(ref[0], got[0], rtol=1e-6, atol=0)
    assert max_rel(ref[1], got[1]) < 1e-6
    assert got[2] < ref[2]


def test_ckpt_same_loss_and_grads_bf16_autocast(base):
    ref = run(copy.deepcopy(base), autocast=True)
    d = copy.deepcopy(base)
    M.apply_opt_in(d, dict(grad_ckpt='all'))
    got = run(d, autocast=True)
    print(f'GC1 bf16 all: loss {float(ref[0]):.9g} vs {float(got[0]):.9g}, rel L2 grad diff {max_rel(ref[1], got[1]):.3g}')
    assert torch.allclose(ref[0], got[0], rtol=1e-3, atol=0)
    assert max_rel(ref[1], got[1]) < 2e-2


def test_bf16_vs_fp32_reference_gap(base):
    """Information for GC1.md (the recipe already trains in bf16 autocast): the size of the bf16-vs-fp32 loss gap."""
    a = run(copy.deepcopy(base))
    b = run(copy.deepcopy(base), autocast=True)
    print(f'GC1 fp32 vs bf16 autocast: loss {float(a[0]):.9g} vs {float(b[0]):.9g}, rel L2 grad diff {max_rel(a[1], b[1]):.3g}')
    assert torch.isfinite(b[0])


def test_channels_last_same_loss(base):
    ref = run(copy.deepcopy(base))
    d = copy.deepcopy(base)
    assert M.apply_opt_in(d, dict(channels_last=1)) == {'channels_last': True}
    got = run(d)
    print(f'GC1 channels_last: loss {float(ref[0]):.9g} vs {float(got[0]):.9g}, rel L2 grad diff {max_rel(ref[1], got[1]):.3g}')
    assert torch.allclose(ref[0], got[0], rtol=1e-5, atol=0)
    assert max_rel(ref[1], got[1]) < 1e-3


def test_freeze_backbone_trains_only_the_decoders(base):
    d = copy.deepcopy(base)
    enc = d.net.encoder
    before = {k: v.detach().clone() for k, v in d.state_dict().items()}
    assert M.apply_opt_in(d, dict(freeze_backbone=1))['freeze_backbone'] > 0
    loss, grads, saved = run(d, tally=True)
    ref = run(copy.deepcopy(base), tally=True)
    enc_names = {'net.model.model.pixel_level_module.encoder.' + n for n, _ in enc.named_parameters()}
    assert enc_names and not (set(grads) & enc_names)          # no backbone gradient
    assert len(grads) > 0 and set(grads) <= {n for n, _ in d.named_parameters()} - enc_names
    print(f'GC1 freeze_backbone: saved activations {ref[2]} -> {saved} B, trainable grads {len(grads)}')
    assert saved < ref[2]
    # backbone in eval during forward (drop_path off): two forwards give identical backbone features
    x = inputs()[0].repeat(1, 3, 1, 1)
    d.train()
    torch.manual_seed(3); f1 = enc(x).feature_maps
    torch.manual_seed(4); f2 = enc(x).feature_maps
    assert all(torch.equal(a, b) for a, b in zip(f1, f2)) and enc.training
    # one optimizer step leaves the backbone weights unchanged
    t = train.Trainer(dict(train.DEFAULTS, df_path='/tmp', min_free_bytes=0), d, list(enc.parameters()), '/tmp',
                      lambda *_: None)
    t.train_step(lambda m, b: m(*b).loss, inputs())
    after = d.state_dict()
    enc_keys = [k for k in before if '.pixel_level_module.encoder.' in k]
    assert enc_keys and all(torch.equal(before[k], after[k]) for k in enc_keys)
    assert any(not torch.equal(before[k], after[k]) for k in before if k not in enc_keys and before[k].is_floating_point())
