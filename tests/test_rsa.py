import numpy as np
import pytest
from scipy.spatial.distance import pdist

from jspace_fmri.rsa import (
    benjamini_hochberg,
    compute_rdm,
    correlate_rdms,
    exact_sign_flip_test,
    reorder_condensed_rdm,
)


def test_compute_and_correlate_identical_rdms():
    features = np.asarray([[1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 3, 2]], dtype=float)
    rdm = compute_rdm(features)
    assert correlate_rdms(rdm, rdm) == pytest.approx(1.0)


def test_reorder_condensed_rdm_matches_direct_computation():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(5, 7))
    old_ids = np.asarray([10, 20, 30, 40, 50])
    new_ids = np.asarray([50, 20, 40])
    reordered = reorder_condensed_rdm(pdist(features), old_ids, new_ids)
    direct = pdist(features[[4, 1, 3]])
    np.testing.assert_allclose(reordered, direct)


def test_exact_sign_flip_all_positive_eight_subjects():
    p = exact_sign_flip_test(np.ones(8), alternative="two-sided")
    assert p == pytest.approx(2 / 256)


def test_benjamini_hochberg_is_monotone_in_rank():
    p = np.asarray([0.04, 0.001, np.nan, 0.03, 0.2])
    adjusted = benjamini_hochberg(p)
    assert np.isnan(adjusted[2])
    assert adjusted[1] <= adjusted[3] <= adjusted[0] <= adjusted[4]
    assert np.all((adjusted[np.isfinite(adjusted)] >= 0) & (adjusted[np.isfinite(adjusted)] <= 1))


def test_constant_feature_row_is_rejected_for_correlation_distance():
    with pytest.raises(ValueError, match="constant"):
        compute_rdm(np.asarray([[1, 1], [1, 2], [2, 1]], dtype=float))

