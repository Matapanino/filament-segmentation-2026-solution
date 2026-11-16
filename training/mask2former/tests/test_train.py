"""Resume restores step/optimizer/scheduler/EMA/RNG exactly; STOP writes a checkpoint and stops; df guard skips writes."""
import json

import numpy as np
import pytest
import torch

import train


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x):
        return self.head(torch.relu(self.bn(self.enc(x))))


def batches(start):
    step = start
    while True:
        g = torch.Generator().manual_seed(1000 + step)
        yield torch.rand(2, 1, 8, 8, generator=g)
        step += 1


def loss_fn(m, x):
    keep = (torch.rand(2, 1, 8, 8) > 0.3).float()  # global RNG: tests RNG restore
    return ((m(x) - x) ** 2 * keep).mean()


def cfg_for(tmp_path, **kw):
    c = dict(train.DEFAULTS, total_steps=6, ckpt_every=3, warmup=2, ema_decay=0.9, log_every=1,
             df_path=str(tmp_path), min_free_bytes=0, debug_stop_after=0)
    c.update(kw)
    return c


def fresh(cfg, run_dir):
    torch.manual_seed(0)
    m = Tiny()
    log = train.Logger(run_dir / 'train.log')
    return train.Trainer(cfg, m, list(m.enc.parameters()), run_dir, log)


def flat_opt(opt):
    out = []
    for s in opt.state_dict()['state'].values():
        for k in sorted(s):
            v = s[k]
            out.append(v.clone() if torch.is_tensor(v) else torch.tensor(float(v)))
    return out


def test_resume_is_exact(tmp_path):
    a_dir = tmp_path / 'a'; a_dir.mkdir()
    b_dir = tmp_path / 'b'; b_dir.mkdir()
    cfg = cfg_for(tmp_path)
    # A: uninterrupted
    torch.manual_seed(123)
    ta = fresh(cfg, a_dir)
    torch.manual_seed(123)
    assert train.run(ta, batches, loss_fn, 6, a_dir / 'STOP', ta.log) == 'done'
    # B: stopped after 4 steps via the STOP file (debug hook), then resumed by a fresh process-like object
    cfg_b = cfg_for(tmp_path, debug_stop_after=4)
    tb = fresh(cfg_b, b_dir)
    torch.manual_seed(123)
    assert train.run(tb, batches, loss_fn, 6, b_dir / 'STOP', tb.log) == 'stopped'
    assert tb.step == 4 and (b_dir / 'ckpt_latest.pt').exists()
    (b_dir / 'STOP').unlink()
    torch.manual_seed(999)  # garbage RNG state: must be overwritten by the checkpoint
    np.random.seed(5)
    tc = fresh(cfg_b, b_dir)
    tc.load()
    assert tc.step == 4
    assert train.run(tc, batches, loss_fn, 6, b_dir / 'STOP', tc.log) == 'done'
    assert tc.step == ta.step == 6
    for (k, va), vc in zip(ta.model.state_dict().items(), tc.model.state_dict().values()):
        assert torch.equal(va, vc), k
    for k in ta.ema:
        assert torch.equal(ta.ema[k], tc.ema[k]), k
    for x, y in zip(flat_opt(ta.opt), flat_opt(tc.opt)):
        assert torch.equal(x, y)
    assert ta.sched.state_dict()['last_epoch'] == tc.sched.state_dict()['last_epoch'] == 6
    assert [g['lr'] for g in ta.opt.param_groups] == [g['lr'] for g in tc.opt.param_groups]


def test_stop_before_first_step_checkpoints_step0(tmp_path):
    cfg = cfg_for(tmp_path)
    t = fresh(cfg, tmp_path)
    (tmp_path / 'STOP').write_text('x')
    assert train.run(t, batches, loss_fn, 6, tmp_path / 'STOP', t.log) == 'stopped'
    assert t.step == 0 and torch.load(tmp_path / 'ckpt_latest.pt', weights_only=False)['step'] == 0


def test_df_guard_skips_checkpoint(tmp_path):
    cfg = cfg_for(tmp_path, min_free_bytes=1 << 62)
    t = fresh(cfg, tmp_path)
    assert train.run(t, batches, loss_fn, 6, tmp_path / 'STOP', t.log) == 'done'
    assert not (tmp_path / 'ckpt_latest.pt').exists()
    rows = [json.loads(x) for x in (tmp_path / 'DF_GUARD_SKIPS.log').read_text().splitlines()]
    assert [r['step'] for r in rows] == [3]


def test_lr_schedule():
    assert train.lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert train.lr_factor(9, 100, 10) == pytest.approx(1.0)
    assert train.lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert train.lr_factor(99, 100, 10) < 1e-3


def test_crop_batches_deterministic_and_cover_files():
    files = [f'f{i}' for i in range(5)]
    img = np.zeros((2048, 2048), np.uint8)
    poly = [[900, 900, 1100, 900, 1100, 950, 900, 950]]
    groups = {f: [[dict(segmentation=poly)], []] for f in files}
    cfg = dict(train.DEFAULTS, batch=3, total_steps=10)
    ds = train.CropBatches({f: img for f in files}, groups, files, cfg)
    a, b = ds[2], ds[2]
    assert np.array_equal(a[0], b[0]) and all(np.array_equal(x, y) for x, y in zip(a[1], b[1])) and a[2] == b[2]
    assert a[0].shape == (3, 512, 512)
    assert sorted(ds.file_for_draw(j) for j in range(5)) == files
    assert sorted(ds.file_for_draw(j) for j in range(5, 10)) == files


def test_accumulation_matches_full_batch(tmp_path):
    def det_loss(m, x):
        return ((m(x) - x) ** 2).mean()

    def full(start):
        step = start
        while True:
            g = torch.Generator().manual_seed(2000 + step)
            yield torch.rand(4, 1, 8, 8, generator=g)
            step += 1

    def halves(start):
        for x in full(start):
            yield [x[:2], x[2:]]
    (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
    ta = fresh(cfg_for(tmp_path, ckpt_every=100), tmp_path / 'a')
    tb = fresh(cfg_for(tmp_path, ckpt_every=100, accum=2), tmp_path / 'b')
    ta.model.eval(); tb.model.eval()  # BatchNorm statistics differ by micro-batch; compare without them
    for m in (ta.model, tb.model):
        m.bn = torch.nn.Identity()
    train.run(ta, full, det_loss, 3, tmp_path / 'a' / 'STOP', ta.log)
    train.run(tb, halves, det_loss, 3, tmp_path / 'b' / 'STOP', tb.log)
    for va, vb in zip(ta.model.parameters(), tb.model.parameters()):
        assert torch.allclose(va, vb, atol=1e-6)
