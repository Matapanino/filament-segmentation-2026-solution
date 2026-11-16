"""The instance-acceptance decision rule (R3; docs/adr/0006).

ADR-0004's image score is

    score = sum(matched Dice) / (n_matched + n_unmatched_gt + n_unmatched_pred)

Adding one predicted component to an image changes the numerator by its Dice
if it matches an otherwise-missed GT filament, and the denominator by +1 only
if it does NOT match (a match trades one ``unmatched_gt`` for one ``matched``,
leaving the denominator alone). With aggregate score ``s`` and mean matched
Dice ``d``, a component with match probability ``p`` is worth

    delta ~ (p * d - (1 - p) * s) / denominator

so it is worth keeping exactly when ``p > s / (s + d)`` — ``break_even`` below.
This is the vision analogue of the tabular harness's prior-correction exponent:
a constant DERIVED from the metric rather than tuned, which made the R3 sweep a
falsifiable prediction instead of a fit (ADR-0006 §2).

**The prediction was falsified and the reason matters** (F-012 §1). The measured
optimum sits at 0.50, not the derived 0.3935, and the classifier is well
calibrated — so the inequality is right but the plug-in was wrong: ``d`` must be
the expected Dice of the MARGINAL component (~0.47 by inversion), not the mean
over all matched components (0.7246). ``break_even`` is therefore kept as the
documented derivation and as a *sanity constant*, while the rule to use is
``keeps(p_match, e_dice, score)`` — ``DiceModel`` estimates ``E[dice]`` per
component and the constant disappears.

``AcceptanceModel``/``DiceModel`` are standardized logistic/ridge fits over the
frozen ``compfeat.FEATURE_NAMES``; both persist as plain coefficients (json) so
the shipped decision rule stays auditable for the rubric's qualitative half.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compfeat import FEATURE_NAMES


def break_even(score: float, dice_mean: float) -> float:
    """Match probability above which keeping a component raises the score."""
    if not 0.0 < dice_mean <= 1.0:
        raise ValueError(f"dice_mean must be in (0,1], got {dice_mean}")
    if score < 0.0:
        raise ValueError(f"score must be >= 0, got {score}")
    return score / (score + dice_mean)


@dataclass
class AcceptanceModel:
    """Standardized logistic regression over ``compfeat.FEATURE_NAMES``."""

    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> "AcceptanceModel":
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features, got {X.shape[1]}"
            )
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        lr = LogisticRegression(C=C, max_iter=2000)
        lr.fit((X - mean) / scale, y)
        return cls(
            names=FEATURE_NAMES,
            mean=mean,
            scale=scale,
            coef=lr.coef_[0].copy(),
            intercept=float(lr.intercept_[0]),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AcceptanceModel":
        return cls(
            names=tuple(d["names"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
        )


@dataclass
class DiceModel:
    """Standardized ridge regression for a component's expected Dice.

    R3 falsified the constant break-even (F-012 §1): the sweep peaked at 0.50
    rather than the derived 0.3935 because the derivation plugged in the mean
    Dice over ALL matched components (0.7246) where the decision needs the
    MARGINAL one's (~0.47 by inversion). Predicting ``E[dice]`` per component
    removes the constant altogether — see ``keeps``.
    """

    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> "DiceModel":
        from sklearn.linear_model import Ridge

        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features, got {X.shape[1]}"
            )
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        rr = Ridge(alpha=alpha)
        rr.fit((X - mean) / scale, np.asarray(y, dtype=np.float64))
        return cls(
            names=FEATURE_NAMES,
            mean=mean,
            scale=scale,
            coef=rr.coef_.copy(),
            intercept=float(rr.intercept_),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Expected Dice, clipped to the [0, 1] the quantity actually lives in."""
        X = np.asarray(X, dtype=np.float64)
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        return np.clip(z, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DiceModel":
        return cls(
            names=tuple(d["names"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
        )


def keeps(
    p_match: np.ndarray, e_dice: np.ndarray, score: float, margin: float = 0.0
) -> np.ndarray:
    """The threshold-free acceptance rule: keep iff the component pays.

    Adding a component moves the ADR-0004 score by ``(p*d - (1-p)*s)/denominator``
    and ``E[dice] = p*d`` is exactly the unconditional expectation the regressor
    estimates, so the test is

        E[dice] - score * (1 - p_match) > margin

    with no constant to tune — the fix F-012 §1 prescribes for the falsified
    ``break_even`` plug-in. ``margin`` (in score-numerator units) buys
    conservatism when the estimates are noisy.
    """
    p_match = np.asarray(p_match, dtype=np.float64)
    e_dice = np.asarray(e_dice, dtype=np.float64)
    if p_match.shape != e_dice.shape:
        raise ValueError(
            f"shape mismatch: p_match {p_match.shape} vs e_dice {e_dice.shape}"
        )
    return e_dice - score * (1.0 - p_match) > margin


def crossfit_predict(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 1.0
) -> tuple[np.ndarray, dict[int, AcceptanceModel]]:
    """Out-of-fold match probabilities: fit on the other folds, predict this one.

    The scoring fold never contributes to the rule that filters it — the same
    cross-fitting discipline the workspace requires of any tuned decision rule.
    """
    X = np.asarray(X, dtype=np.float64)
    folds = np.asarray(folds)
    p = np.full(len(X), np.nan, dtype=np.float64)
    models: dict[int, AcceptanceModel] = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            raise ValueError(f"fold {f} leaves an empty train or test side")
        m = AcceptanceModel.fit(X[tr], y[tr], C=C)
        models[int(f)] = m
        p[te] = m.predict_proba(X[te])
    if np.isnan(p).any():
        raise AssertionError("some components got no out-of-fold prediction")
    return p, models


def crossfit_dice(
    X: np.ndarray, y_dice: np.ndarray, folds: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, dict[int, DiceModel]]:
    """Out-of-fold expected-Dice predictions, same discipline as the classifier."""
    X = np.asarray(X, dtype=np.float64)
    folds = np.asarray(folds)
    e = np.full(len(X), np.nan, dtype=np.float64)
    models: dict[int, DiceModel] = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            raise ValueError(f"fold {f} leaves an empty train or test side")
        m = DiceModel.fit(X[tr], y_dice[tr], alpha=alpha)
        models[int(f)] = m
        e[te] = m.predict(X[te])
    if np.isnan(e).any():
        raise AssertionError("some components got no out-of-fold prediction")
    return e, models
