"""Small, explicit RSA and participant-level inference utilities."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ttest_1samp


def expected_condensed_length(n_items: int) -> int:
    if n_items < 2:
        raise ValueError("an RDM needs at least two items")
    return n_items * (n_items - 1) // 2


def validate_ids(ids: np.ndarray, *, name: str = "ids") -> np.ndarray:
    ids = np.asarray(ids).reshape(-1)
    if ids.size < 3:
        raise ValueError(f"{name} must contain at least three items")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{name} contains duplicates")
    return ids


def compute_rdm(features: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Return a finite condensed RDM for `[items, features]` input."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if features.shape[0] < 3:
        raise ValueError("at least three items are required")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")
    if metric == "correlation" and np.any(np.std(features, axis=1) == 0):
        raise ValueError("correlation distance is undefined for constant feature rows")
    rdm = pdist(features, metric=metric)
    if not np.all(np.isfinite(rdm)):
        raise ValueError("computed RDM contains non-finite values")
    return rdm.astype(np.float64, copy=False)


def validate_rdm(rdm: np.ndarray, n_items: int | None = None) -> np.ndarray:
    rdm = np.asarray(rdm, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(rdm)):
        raise ValueError("RDM contains non-finite values")
    if n_items is not None and rdm.size != expected_condensed_length(n_items):
        raise ValueError(
            f"RDM has {rdm.size} entries; expected {expected_condensed_length(n_items)} "
            f"for {n_items} items"
        )
    if np.std(rdm) == 0:
        raise ValueError("RDM is constant, so RSA correlation is undefined")
    return rdm


def correlate_rdms(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    a = validate_rdm(a)
    b = validate_rdm(b)
    if a.shape != b.shape:
        raise ValueError(f"RDM shapes differ: {a.shape} vs {b.shape}")
    if method == "pearson":
        return float(pearsonr(a, b).statistic)
    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    raise ValueError(f"unknown RSA correlation method: {method}")


def reorder_condensed_rdm(
    rdm: np.ndarray, old_ids: np.ndarray, new_ids: np.ndarray
) -> np.ndarray:
    """Reorder/subset a condensed RDM from `old_ids` to `new_ids`."""
    old_ids = validate_ids(old_ids, name="old_ids")
    new_ids = validate_ids(new_ids, name="new_ids")
    rdm = validate_rdm(rdm, old_ids.size)
    position = {item: i for i, item in enumerate(old_ids.tolist())}
    missing = [item for item in new_ids.tolist() if item not in position]
    if missing:
        raise ValueError(f"RDM is missing {len(missing)} requested IDs; first: {missing[:5]}")
    order = np.asarray([position[item] for item in new_ids.tolist()], dtype=int)
    full = squareform(rdm)
    return squareform(full[np.ix_(order, order)], checks=False)


def exact_sign_flip_test(values: np.ndarray, alternative: str = "two-sided") -> float:
    """Exact paired sign-flip test on the participant-level mean contrast.

    With the intended eight NSD participants this enumerates all 256 sign
    assignments. The observed assignment is included, so no ad-hoc +1
    correction is needed.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size > 20:
        raise ValueError("exact enumeration is intentionally limited to 20 observations")
    observed = float(np.mean(values))
    permuted = np.fromiter(
        (np.mean(values * np.asarray(signs)) for signs in product((-1.0, 1.0), repeat=values.size)),
        dtype=np.float64,
        count=2**values.size,
    )
    tolerance = np.finfo(float).eps * 32
    if alternative == "two-sided":
        return float(np.mean(np.abs(permuted) >= abs(observed) - tolerance))
    if alternative == "greater":
        return float(np.mean(permuted >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(permuted <= observed + tolerance))
    raise ValueError(f"unknown alternative: {alternative}")


def one_sample_t(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    result = ttest_1samp(values, popmean=0.0, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs and order."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]
    if p.size == 0:
        return adjusted
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p)
    ranked = p[order]
    raw_adjusted = ranked * p.size / np.arange(1, p.size + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def leave_one_subject_out_noise_ceilings(
    subject_rdms: dict[str, np.ndarray], method: str = "pearson"
) -> dict[str, float]:
    """Correlate each participant RDM with the mean RDM of the others."""
    if len(subject_rdms) < 3:
        raise ValueError("noise ceilings require at least three participants")
    names = sorted(subject_rdms)
    arrays = {name: validate_rdm(subject_rdms[name]) for name in names}
    lengths = {arr.size for arr in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("participant RDM lengths differ")
    output: dict[str, float] = {}
    for name in names:
        others = np.stack([arrays[other] for other in names if other != name])
        output[name] = correlate_rdms(arrays[name], others.mean(axis=0), method=method)
    return output

