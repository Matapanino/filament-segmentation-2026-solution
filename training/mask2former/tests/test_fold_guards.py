"""Fold guards (added after a code review, 2026-11-14): eval/export must use the model's training fold; resume must not change it."""
import pytest

import eval_ckpt
import train


def test_eval_fold_must_match_training_fold():
    eval_ckpt.check_fold({'fold': 2}, 2)
    with pytest.raises(SystemExit):
        eval_ckpt.check_fold({'fold': 0}, 2)


def test_eval_fold_shipped_cfg_without_fold_passes():
    eval_ckpt.check_fold({'input_scale': 0.5, 'shipped': 'MF2'}, 2)


def test_resume_rejects_other_fold():
    train.check_resume_config({'fold': 1, 'seed': 42}, {'fold': 1, 'seed': 42})
    with pytest.raises(SystemExit):
        train.check_resume_config({'fold': 0}, {'fold': 1})
