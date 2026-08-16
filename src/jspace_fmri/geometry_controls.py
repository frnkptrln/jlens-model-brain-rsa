"""Utilities and analysis for the frozen dense-transport geometry controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rsa import benjamini_hochberg, correlate_rdms, exact_sign_flip_test, validate_rdm

PRIMARY_LAYERS = (18, 27)
PRIMARY_ROIS = ("ventral", "lateral", "parietal")
BRANCHES = ("final_gain", "no_final_norm")
REPRESENTATIONS = ("raw", "actual", "polar")


def centered_unit(vector: np.ndarray) -> np.ndarray:
    """Return the centered unit vector used by Pearson correlation."""
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("vector contains non-finite values")
    values = values - values.mean()
    norm = float(np.linalg.norm(values))
    if norm == 0:
        raise ValueError("centered vector has zero norm")
    return values / norm


def deterministic_permutation(
    width: int,
    *,
    master_seed: int,
    control_index: int,
    layer: int,
    side: int,
) -> np.ndarray:
    """Generate one frozen input/output coordinate permutation."""
    if width < 2:
        raise ValueError("width must be at least two")
    if control_index < 0:
        raise ValueError("control_index must be non-negative")
    if side not in (0, 1):
        raise ValueError("side must be 0 (input) or 1 (output)")
    generator = np.random.default_rng(
        np.random.SeedSequence([master_seed, control_index, layer, side])
    )
    return generator.permutation(width).astype(np.int64, copy=False)


def permuted_transport_numpy(
    residuals: np.ndarray,
    jacobian: np.ndarray,
    input_permutation: np.ndarray,
    output_permutation: np.ndarray,
) -> np.ndarray:
    """Apply ``P_out J P_in h`` using the row-vector convention of jlens."""
    residuals = np.asarray(residuals)
    jacobian = np.asarray(jacobian)
    input_permutation = np.asarray(input_permutation, dtype=np.int64)
    output_permutation = np.asarray(output_permutation, dtype=np.int64)
    if residuals.ndim != 2 or jacobian.ndim != 2:
        raise ValueError("residuals and jacobian must both be two-dimensional")
    width = residuals.shape[1]
    if jacobian.shape != (width, width):
        raise ValueError("jacobian shape does not match residual width")
    expected = np.arange(width)
    if not np.array_equal(np.sort(input_permutation), expected):
        raise ValueError("input_permutation is not a permutation")
    if not np.array_equal(np.sort(output_permutation), expected):
        raise ValueError("output_permutation is not a permutation")
    return (residuals[:, input_permutation] @ jacobian.T)[:, output_permutation]


def polar_factor_numpy(jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the orthogonal polar factor ``U V^T`` and singular values."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("jacobian must be square")
    left, singular_values, right_t = np.linalg.svd(jacobian, full_matrices=False)
    return left @ right_t, singular_values


def build_primary_neural_targets(
    neural_dir: Path, output_path: Path | None = None
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compress the three-ROI primary target into one exact target per subject.

    For a model RDM ``m``, ``centered_unit(m) @ target[subject]`` equals the
    arithmetic mean of its three ROI-wise Pearson RSA scores, up to storage
    precision. The returned targets are float32 to keep the remote payload
    compact; validation against direct Pearson scores is performed separately.
    """
    paths = sorted(neural_dir.glob("subj*.npz"))
    if len(paths) != 8:
        raise ValueError(f"expected eight participant files, found {len(paths)}")
    canonical_ids: np.ndarray | None = None
    subjects: list[str] = []
    targets: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["nsd_ids"], dtype=np.int64)
            if canonical_ids is None:
                canonical_ids = ids
            elif not np.array_equal(ids, canonical_ids):
                raise ValueError(f"stimulus order differs in {path}")
            roi_units = []
            for roi in PRIMARY_ROIS:
                key = f"rdm__{roi}"
                if key not in archive:
                    raise ValueError(f"{path} is missing {key}")
                roi_units.append(centered_unit(archive[key]))
        subjects.append(path.stem)
        targets.append(np.mean(roi_units, axis=0).astype(np.float32))
    assert canonical_ids is not None
    target_array = np.stack(targets)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            nsd_ids=canonical_ids,
            subjects=np.asarray(subjects),
            primary_targets=target_array,
            rois=np.asarray(PRIMARY_ROIS),
        )
    return canonical_ids, subjects, target_array


def score_primary_targets(rdm: np.ndarray, targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError("targets must be [subjects, RDM entries]")
    model_unit = centered_unit(rdm)
    if targets.shape[1] != model_unit.size:
        raise ValueError("target and model RDM lengths differ")
    return targets @ model_unit


def parse_geometry_key(key: str) -> tuple[str, str, str]:
    parts = key.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"geometry key {key!r} must be REPRESENTATION__BRANCH__LOCATION"
        )
    representation, branch, location = parts
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown geometry representation {representation!r}")
    if branch not in BRANCHES:
        raise ValueError(f"unknown preprocessing branch {branch!r}")
    if not location.startswith("layer_"):
        raise ValueError(f"invalid layer label {location!r}")
    return representation, branch, location


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(values, ddof=1))
    return {
        "n_subjects": int(values.size),
        "mean": float(np.mean(values)),
        "sem": float(standard_deviation / np.sqrt(values.size)),
        "cohens_dz": (
            float(np.mean(values) / standard_deviation)
            if standard_deviation > 0
            else float("nan")
        ),
        "p_signflip_two_sided": exact_sign_flip_test(values, alternative="two-sided"),
        "p_signflip_greater": exact_sign_flip_test(values, alternative="greater"),
    }


def analyse_geometry_controls(
    artifact_path: Path,
    neural_dir: Path,
    output_dir: Path,
    *,
    previous_rdms_path: Path | None = None,
    rdm_tolerance: float = 5e-5,
    correlation_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Analyse a validated remote geometry-control artifact."""
    with np.load(artifact_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["nsd_ids"], dtype=np.int64)
        rdms = {
            key.removeprefix("rdm__"): validate_rdm(archive[key], ids.size)
            for key in archive.files
            if key.startswith("rdm__")
        }
        manifest = (
            json.loads(str(archive["manifest_json"]))
            if "manifest_json" in archive
            else {}
        )
        control_subjects = (
            np.asarray(archive["control_subjects"]).astype(str).tolist()
            if "control_subjects" in archive
            else []
        )
        control_primary = {
            branch: np.asarray(archive[f"control_primary_rsa__{branch}"], dtype=float)
            for branch in BRANCHES
            if f"control_primary_rsa__{branch}" in archive
        }
    required = {
        f"{representation}__{branch}__layer_{layer:02d}"
        for representation in REPRESENTATIONS
        for branch in BRANCHES
        for layer in PRIMARY_LAYERS
    }
    missing = sorted(required - set(rdms))
    if missing:
        raise ValueError(f"geometry artifact is missing RDMs: {missing}")
    parsed = {key: parse_geometry_key(key) for key in rdms}

    score_rows: list[dict[str, object]] = []
    neural_paths = sorted(neural_dir.glob("subj*.npz"))
    if len(neural_paths) != 8:
        raise ValueError(f"expected eight participant files, found {len(neural_paths)}")
    for path in neural_paths:
        with np.load(path, allow_pickle=False) as neural:
            if not np.array_equal(neural["nsd_ids"], ids):
                raise ValueError(f"stimulus order differs in {path}")
            for roi in PRIMARY_ROIS:
                neural_rdm = validate_rdm(neural[f"rdm__{roi}"], ids.size)
                for key, model_rdm in rdms.items():
                    representation, branch, location = parsed[key]
                    score_rows.append(
                        {
                            "subject": path.stem,
                            "roi": roi,
                            "representation": representation,
                            "branch": branch,
                            "location": location,
                            "rsa": correlate_rdms(model_rdm, neural_rdm),
                        }
                    )
    scores = pd.DataFrame(score_rows)
    wide = scores.pivot(
        index=["subject", "roi", "branch", "location"],
        columns="representation",
        values="rsa",
    ).reset_index()
    for name, left, right in (
        ("actual_minus_raw", "actual", "raw"),
        ("polar_minus_actual", "polar", "actual"),
        ("polar_minus_raw", "polar", "raw"),
    ):
        wide[name] = wide[left] - wide[right]

    cell_rows: list[dict[str, object]] = []
    for (branch, location, roi), frame in wide.groupby(
        ["branch", "location", "roi"], sort=True
    ):
        for contrast in ("actual_minus_raw", "polar_minus_actual", "polar_minus_raw"):
            cell_rows.append(
                {
                    "branch": branch,
                    "location": location,
                    "roi": roi,
                    "contrast": contrast,
                    **_summary(frame[contrast].to_numpy()),
                }
            )
    cells = pd.DataFrame(cell_rows)
    cells["p_signflip_fdr_6_cells"] = np.nan
    for (_branch, _contrast), indices in cells.groupby(["branch", "contrast"]).groups.items():
        cells.loc[indices, "p_signflip_fdr_6_cells"] = benjamini_hochberg(
            cells.loc[indices, "p_signflip_two_sided"].to_numpy()
        )

    participant = (
        wide.groupby(["subject", "branch"], as_index=False)
        .agg(
            rsa_raw=("raw", "mean"),
            rsa_actual=("actual", "mean"),
            rsa_polar=("polar", "mean"),
            actual_minus_raw=("actual_minus_raw", "mean"),
            polar_minus_actual=("polar_minus_actual", "mean"),
            polar_minus_raw=("polar_minus_raw", "mean"),
        )
        .sort_values(["branch", "subject"])
    )
    primary_values = participant.loc[
        participant["branch"] == "final_gain", "polar_minus_actual"
    ].to_numpy()
    primary = {
        "test": "polar-minus-actual RSA averaged across 2 layers x 3 ROIs",
        "branch": "final_gain",
        "alternative": "greater",
        **_summary(primary_values),
    }

    model_rows = []
    for branch in BRANCHES:
        for layer in PRIMARY_LAYERS:
            location = f"layer_{layer:02d}"
            keys = {
                representation: f"{representation}__{branch}__{location}"
                for representation in REPRESENTATIONS
            }
            for first, second in (("raw", "actual"), ("raw", "polar"), ("actual", "polar")):
                model_rows.append(
                    {
                        "branch": branch,
                        "location": location,
                        "first": first,
                        "second": second,
                        "rdm_pearson": correlate_rdms(rdms[keys[first]], rdms[keys[second]]),
                    }
                )
    model_correlations = pd.DataFrame(model_rows)

    reproduction_rows: list[dict[str, object]] = []
    if previous_rdms_path is not None:
        with np.load(previous_rdms_path, allow_pickle=False) as previous:
            if not np.array_equal(previous["nsd_ids"], ids):
                raise ValueError("previous and new model RDM stimulus orders differ")
            for branch in BRANCHES:
                old_prompt = "raw" if branch == "final_gain" else "raw_nofinalnorm"
                for layer in PRIMARY_LAYERS:
                    for representation, old_kind in (("raw", "raw"), ("actual", "jlens")):
                        new_key = f"{representation}__{branch}__layer_{layer:02d}"
                        old_key = f"rdm__{old_kind}__{old_prompt}__layer_{layer:02d}"
                        if old_key not in previous:
                            raise ValueError(f"previous artifact is missing {old_key}")
                        old = validate_rdm(previous[old_key], ids.size)
                        maximum = float(np.max(np.abs(rdms[new_key] - old)))
                        correlation = correlate_rdms(rdms[new_key], old)
                        reproduction_rows.append(
                            {
                                "branch": branch,
                                "location": f"layer_{layer:02d}",
                                "representation": representation,
                                "max_abs_difference": maximum,
                                "rdm_pearson": correlation,
                                "pass": bool(
                                    maximum <= rdm_tolerance
                                    and correlation >= 1.0 - correlation_tolerance
                                ),
                            }
                        )
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction_passed = bool(reproduction.empty or reproduction["pass"].all())

    control_rows: list[dict[str, object]] = []
    if control_primary:
        expected_subjects = participant.loc[
            participant["branch"] == "final_gain", "subject"
        ].tolist()
        if control_subjects != expected_subjects:
            raise ValueError("coordinate-control participant order differs")
        for branch, values in control_primary.items():
            if values.ndim != 2 or values.shape[1] != len(control_subjects):
                raise ValueError(f"invalid coordinate-control array for {branch}: {values.shape}")
            actual_by_subject = participant.loc[
                participant["branch"] == branch, "rsa_actual"
            ].to_numpy()
            raw_by_subject = participant.loc[
                participant["branch"] == branch, "rsa_raw"
            ].to_numpy()
            for index, row in enumerate(values):
                control_rows.append(
                    {
                        "branch": branch,
                        "control_index": index,
                        "mean_rsa": float(np.mean(row)),
                        "mean_delta_vs_raw": float(np.mean(row - raw_by_subject)),
                        "mean_actual_minus_control": float(np.mean(actual_by_subject - row)),
                    }
                )
    controls = pd.DataFrame(control_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "rsa_scores.csv", index=False)
    wide.to_csv(output_dir / "participant_cell_contrasts.csv", index=False)
    participant.to_csv(output_dir / "primary_participant_contrasts.csv", index=False)
    cells.to_csv(output_dir / "cell_contrasts.csv", index=False)
    model_correlations.to_csv(output_dir / "model_rdm_correlations.csv", index=False)
    reproduction.to_csv(output_dir / "reproduction_checks.csv", index=False)
    controls.to_csv(output_dir / "coordinate_permutation_controls.csv", index=False)
    with (output_dir / "primary_test.json").open("w", encoding="utf-8") as handle:
        json.dump(primary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "prospective post-pilot dense-transport diagnostic",
        "artifact": str(artifact_path),
        "artifact_manifest": manifest,
        "n_images": int(ids.size),
        "n_subjects": int(scores["subject"].nunique()),
        "primary_test": primary,
        "reproduction_checks_passed": reproduction_passed,
        "n_coordinate_controls": int(
            controls["control_index"].nunique() if not controls.empty else 0
        ),
        "claim_boundary": (
            "Dense Jacobian transport only; not formal sparse non-negative J-space "
            "and not evidence for a homologous neural workspace."
        ),
    }
    if not controls.empty:
        final_controls = controls[controls["branch"] == "final_gain"]
        actual_mean = float(
            participant.loc[participant["branch"] == "final_gain", "rsa_actual"].mean()
        )
        summary["actual_coordinate_reference_percentile"] = float(
            100.0 * np.mean(final_controls["mean_rsa"] <= actual_mean)
        )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not reproduction_passed:
        raise RuntimeError("new raw/actual RDMs failed the frozen reproduction tolerance")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    targets = subparsers.add_parser("build-targets")
    targets.add_argument("--neural-dir", type=Path, required=True)
    targets.add_argument("--output", type=Path, required=True)
    analyse = subparsers.add_parser("analyse")
    analyse.add_argument("--artifact", type=Path, required=True)
    analyse.add_argument("--neural-dir", type=Path, required=True)
    analyse.add_argument("--output-dir", type=Path, required=True)
    analyse.add_argument("--previous-rdms", type=Path)
    args = parser.parse_args()
    if args.command == "build-targets":
        ids, subjects, target_array = build_primary_neural_targets(
            args.neural_dir, args.output
        )
        print(
            json.dumps(
                {
                    "n_images": int(ids.size),
                    "subjects": subjects,
                    "target_shape": list(target_array.shape),
                },
                indent=2,
            )
        )
    else:
        summary = analyse_geometry_controls(
            args.artifact,
            args.neural_dir,
            args.output_dir,
            previous_rdms_path=args.previous_rdms,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

