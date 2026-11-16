"""GC1: opt-in memory/speed flags (--grad-ckpt, --tf32, --channels-last) are default off, and off = byte-identical.

The reference hashes are sha256(json.dumps(cfg)) of train.py before GC1 (md5 2a70d142..., 2026-11-14 20:37 JST).
"""
import hashlib
import json

import pytest
import torch

import train

BASE = ['--run-dir', 'r', '--data-root', 'd', '--fold-table', 'f']
REF = {(): 'b0c95111e094ae4451e060be2612e2e332ccd08c41a77dcbf938108f57f9747b',
       ('--fold', 'all'): 'd8df01b493da8f9c51b1c33338cd64b4be7ce496f5ea7ac791180134d5f27ebd',
       ('--crop', '1536', '--accum', '8', '--num-queries', '50'):
           '2958a00c378c87c435ad690347110c0c1a19cfb487207d46cd3319dfd8e7a30f'}


def cfg_of(*extra):
    return train.config_from_args(train.build_parser().parse_args(BASE + list(extra)))


@pytest.mark.parametrize('extra', list(REF))
def test_default_cfg_is_byte_identical(extra):
    assert hashlib.sha256(json.dumps(cfg_of(*extra)).encode()).hexdigest() == REF[extra]


@pytest.mark.parametrize('extra', [('--grad-ckpt', ''), ('--tf32', '0'), ('--channels-last', '0')])
def test_explicit_off_is_byte_identical(extra):
    assert hashlib.sha256(json.dumps(cfg_of(*extra)).encode()).hexdigest() == REF[()]


def test_flags_on_are_recorded():
    assert cfg_of('--grad-ckpt')['grad_ckpt'] == 'all'
    assert cfg_of('--grad-ckpt', 'backbone')['grad_ckpt'] == 'backbone'
    assert cfg_of('--grad-ckpt', 'backbone,pixel')['grad_ckpt'] == 'backbone,pixel'
    c = cfg_of('--tf32', '1', '--channels-last', '1')
    assert c['tf32'] == 1 and c['channels_last'] == 1 and 'grad_ckpt' not in c


def test_bad_ckpt_part_refused():
    with pytest.raises(SystemExit):
        cfg_of('--grad-ckpt', 'neck')


def test_parse_ckpt_parts():
    import model
    assert model.ckpt_parts('all') == ('backbone', 'pixel', 'decoder')
    assert model.ckpt_parts('decoder,backbone') == ('backbone', 'decoder')
    with pytest.raises(ValueError):
        model.ckpt_parts('neck')


def test_fused_adam_flag_matches_default_optimizer(tmp_path):
    """--fused-adam 1 = AdamW(fused=True) + foreach clip: same math as the default (foreach=False) up to fp rounding."""
    from test_train import Tiny, batches, loss_fn
    out = []
    for fused in (0, 1):
        cfg = dict(train.DEFAULTS, total_steps=4, warmup=1, df_path=str(tmp_path), min_free_bytes=0)
        if fused:
            cfg['fused_adam'] = 1
        torch.manual_seed(0)
        m = Tiny()
        t = train.Trainer(cfg, m, list(m.enc.parameters()), tmp_path, train.Logger(tmp_path / 'log'))
        assert t.opt.param_groups[0]['fused'] is (True if fused else None)
        torch.manual_seed(3)
        it = batches(0)
        for _ in range(4):
            t.train_step(loss_fn, next(it))
        out.append(torch.cat([p.detach().flatten() for p in m.parameters()]))
    assert torch.allclose(out[0], out[1], rtol=1e-5, atol=1e-7)
