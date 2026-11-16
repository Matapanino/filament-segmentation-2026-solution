"""Instancing-rule override (order PP1): `--rule thr=..,min_area=..,close_px=..,link_gap=..,link_angle=..`.

parse_rule(None or '') returns the SHIP_R3 object itself (no flag = chain2 exactly). Keys not given keep their SHIP_R3
value. Only these five Rule fields can be overridden; everything else (link_window, link_min_area, hysteresis, ...) stays.
"""
import math
from filseg.instancing import SHIP_R3

KEYS = {'thr': float, 'min_area': int, 'close_px': int, 'link_gap': int, 'link_angle': float}
ORDER = ('thr', 'min_area', 'close_px', 'link_gap', 'link_angle')


def _num(key, s):
    if KEYS[key] is int:
        if not s.strip().lstrip('-').isdigit(): raise ValueError(f'{key} must be an integer: {s!r}')
        v = int(s)
    else:
        try: v = float(s)
        except ValueError: raise ValueError(f'{key} must be a number: {s!r}') from None
        if not math.isfinite(v): raise ValueError(f'{key} must be finite: {s!r}')
    if key == 'thr' and not 0 < v < 1: raise ValueError(f'thr must be in (0, 1): {v}')
    if key == 'min_area' and v < 1: raise ValueError(f'min_area must be >= 1: {v}')
    if key in ('close_px', 'link_gap') and v < 0: raise ValueError(f'{key} must be >= 0: {v}')
    if key == 'link_angle' and not 0 < v <= 90: raise ValueError(f'link_angle must be in (0, 90]: {v}')
    return v


def parse_rule(text, base=SHIP_R3):
    if text is None or not text.strip(): return base
    kw = {}
    for part in text.split(','):
        if '=' not in part: raise ValueError(f'expected key=value: {part!r}')
        k, v = (x.strip() for x in part.split('=', 1))
        if k not in KEYS: raise ValueError(f'unknown rule key {k!r} (allowed: {", ".join(ORDER)})')
        if k in kw: raise ValueError(f'duplicate rule key {k!r}')
        if not v: raise ValueError(f'empty value for {k!r}')
        kw[k] = _num(k, v)
    return base.with_(**kw)


def is_ship(rule): return rule == SHIP_R3


def _fmt(v): return f'{v:g}' if isinstance(v, float) else str(v)


def rule_text(rule): return ','.join(f'{k}={_fmt(getattr(rule, k))}' for k in ORDER)


def rule_tag(rule):
    """Short label: 'r3' for SHIP_R3, else the changed fields, e.g. t45 (thr .45), a128, c31, g90, n30 (link angle)."""
    if is_ship(rule): return 'r3'
    code = dict(thr='t', min_area='a', close_px='c', link_gap='g', link_angle='n')
    out = ''
    for k in ORDER:
        v = getattr(rule, k)
        if v != getattr(SHIP_R3, k): out += code[k] + (f'{round(v * 100):02d}' if k == 'thr' else _fmt(v))
    return out


def rule_spec(rule):
    """The chain_inputs.json form: canonical text + the five fields (what worker.set_rule and csv_worker read)."""
    return dict(text=rule_text(rule), fields={k: getattr(rule, k) for k in ORDER})
