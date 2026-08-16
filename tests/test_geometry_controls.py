import numpy as np
from scipy.stats import pearsonr

from jspace_fmri.geometry_controls import (
    centered_unit,
    deterministic_permutation,
    permuted_transport_numpy,
    polar_factor_numpy,
    score_primary_targets,
)


def test_coordinate_permutation_preserves_singular_values_and_matches_matrix():
    rng = np.random.default_rng(7)
    jacobian = rng.normal(size=(7, 7))
    residuals = rng.normal(size=(5, 7))
    input_permutation = deterministic_permutation(
        7, master_seed=20260816, control_index=4, layer=18, side=0
    )
    output_permutation = deterministic_permutation(
        7, master_seed=20260816, control_index=4, layer=18, side=1
    )
    input_matrix = np.eye(7)[input_permutation]
    output_matrix = np.eye(7)[output_permutation]
    controlled_jacobian = output_matrix @ jacobian @ input_matrix
    observed = permuted_transport_numpy(
        residuals, jacobian, input_permutation, output_permutation
    )
    np.testing.assert_allclose(observed, residuals @ controlled_jacobian.T)
    np.testing.assert_allclose(
        np.linalg.svd(controlled_jacobian, compute_uv=False),
        np.linalg.svd(jacobian, compute_uv=False),
        atol=1e-12,
    )


def test_polar_factor_is_orthogonal_and_flattens_singular_values():
    rng = np.random.default_rng(8)
    jacobian = rng.normal(size=(9, 9)) + np.eye(9)
    polar, singular_values = polar_factor_numpy(jacobian)
    np.testing.assert_allclose(polar.T @ polar, np.eye(9), atol=1e-12)
    np.testing.assert_allclose(np.linalg.svd(polar, compute_uv=False), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        singular_values, np.linalg.svd(jacobian, compute_uv=False), atol=1e-12
    )


def test_compressed_primary_target_equals_mean_roi_pearson():
    rng = np.random.default_rng(9)
    model = rng.normal(size=100)
    rois = rng.normal(size=(3, 100))
    target = np.mean([centered_unit(row) for row in rois], axis=0)[None, :]
    observed = score_primary_targets(model, target)[0]
    expected = np.mean([pearsonr(model, row).statistic for row in rois])
    np.testing.assert_allclose(observed, expected, atol=1e-12)
