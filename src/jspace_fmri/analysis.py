"""Run focused ROI RSA and paired participant-level J-lens contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rsa import (
    benjamini_hochberg,
    compute_rdm,
    correlate_rdms,
    exact_sign_flip_test,
    leave_one_subject_out_noise_ceilings,
    one_sample_t,
    reorder_condensed_rdm,
    validate_ids,
    validate_rdm,
)

RESERVED_MODEL_KEYS = {
    "nsd_ids",
    "coco_ids",
    "caption_indices",
    "manifest_json",
}


def parse_representation_key(key: str) -> tuple[str, str, str]:
    parts = key.split("__", 2)
    if len(parts) != 3:
        raise ValueError(
            f"model feature key {key!r} must be KIND__PROMPT__LOCATION, "
            "for example jlens__raw__layer_09"
        )
    return parts[0], parts[1], parts[2]


def load_model_features(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if "nsd_ids" not in archive:
            raise ValueError(f"{path} has no nsd_ids array")
        ids = validate_ids(archive["nsd_ids"], name="model nsd_ids")
        features = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in RESERVED_MODEL_KEYS
        }
    if not features:
        raise ValueError(f"{path} contains no model feature arrays")
    for key, values in features.items():
        parse_representation_key(key)
        if values.ndim != 2 or values.shape[0] != ids.size:
            raise ValueError(f"{key} has shape {values.shape}; expected [{ids.size}, dimensions]")
    return ids, features


def load_and_merge_model_features(
    paths: Path | list[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load one or more feature archives and align them to the first archive."""
    feature_paths = [paths] if isinstance(paths, Path) else list(paths)
    if not feature_paths:
        raise ValueError("at least one model feature archive is required")
    canonical_ids, merged = load_model_features(feature_paths[0])
    canonical_position = {item: index for index, item in enumerate(canonical_ids.tolist())}
    for path in feature_paths[1:]:
        ids, features = load_model_features(path)
        if set(ids.tolist()) != set(canonical_ids.tolist()):
            missing = sorted(set(canonical_ids.tolist()) - set(ids.tolist()))
            extra = sorted(set(ids.tolist()) - set(canonical_ids.tolist()))
            raise ValueError(
                f"{path} stimulus IDs differ from the first archive; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        duplicate_keys = sorted(set(merged) & set(features))
        if duplicate_keys:
            raise ValueError(f"duplicate feature keys across archives: {duplicate_keys}")
        source_position = {item: index for index, item in enumerate(ids.tolist())}
        order = np.asarray(
            [source_position[item] for item in canonical_position], dtype=int
        )
        merged.update({key: values[order] for key, values in features.items()})
    return canonical_ids, merged


def load_model_rdms(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load condensed model RDMs saved as ``rdm__KIND__PROMPT__LOCATION``."""
    with np.load(path, allow_pickle=False) as archive:
        if "nsd_ids" not in archive:
            raise ValueError(f"{path} has no nsd_ids array")
        ids = validate_ids(archive["nsd_ids"], name="model RDM nsd_ids")
        rdms: dict[str, np.ndarray] = {}
        for archive_key in archive.files:
            if archive_key in RESERVED_MODEL_KEYS:
                continue
            if not archive_key.startswith("rdm__"):
                raise ValueError(
                    f"precomputed model RDM key {archive_key!r} must start with 'rdm__'"
                )
            key = archive_key.removeprefix("rdm__")
            parse_representation_key(key)
            if key in rdms:
                raise ValueError(f"duplicate model RDM key {key!r} in {path}")
            rdms[key] = validate_rdm(archive[archive_key], ids.size)
    if not rdms:
        raise ValueError(f"{path} contains no precomputed model RDMs")
    return ids, rdms


def load_and_merge_model_rdms(
    paths: Path | list[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rdm_paths = [paths] if isinstance(paths, Path) else list(paths)
    if not rdm_paths:
        raise ValueError("at least one model RDM archive is required")
    canonical_ids, merged = load_model_rdms(rdm_paths[0])
    for path in rdm_paths[1:]:
        ids, rdms = load_model_rdms(path)
        if set(ids.tolist()) != set(canonical_ids.tolist()):
            missing = sorted(set(canonical_ids.tolist()) - set(ids.tolist()))
            extra = sorted(set(ids.tolist()) - set(canonical_ids.tolist()))
            raise ValueError(
                f"{path} stimulus IDs differ from the first archive; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        duplicate_keys = sorted(set(merged) & set(rdms))
        if duplicate_keys:
            raise ValueError(f"duplicate RDM keys across archives: {duplicate_keys}")
        merged.update(
            {
                key: reorder_condensed_rdm(rdm, ids, canonical_ids)
                for key, rdm in rdms.items()
            }
        )
    return canonical_ids, merged


def load_neural_file(
    path: Path, canonical_ids: np.ndarray, *, require_complete: bool
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "nsd_ids" not in archive:
            raise ValueError(f"{path} has no nsd_ids")
        neural_ids = validate_ids(archive["nsd_ids"], name=f"{path.name} nsd_ids")
        common = np.asarray([item for item in canonical_ids if item in set(neural_ids.tolist())])
        if require_complete and common.size != canonical_ids.size:
            raise ValueError(
                f"{path} covers {common.size}/{canonical_ids.size} model stimuli; "
                "omit --require-complete only for an explicitly planned subset analysis"
            )
        neural_position = {item: i for i, item in enumerate(neural_ids.tolist())}
        order = np.asarray([neural_position[item] for item in common.tolist()], dtype=int)
        rdms: dict[str, np.ndarray] = {}
        for key in archive.files:
            if key in {"nsd_ids", "metadata_json"}:
                continue
            if key.startswith("patterns__"):
                roi = key.removeprefix("patterns__")
                if roi in rdms:
                    raise ValueError(f"duplicate neural representation for ROI {roi} in {path}")
                patterns = np.asarray(archive[key])
                if patterns.ndim != 2 or patterns.shape[0] != neural_ids.size:
                    raise ValueError(f"{path}:{key} has incompatible shape {patterns.shape}")
                rdms[roi] = compute_rdm(patterns[order], metric="correlation")
            elif key.startswith("rdm__"):
                roi = key.removeprefix("rdm__")
                if roi in rdms:
                    raise ValueError(f"duplicate neural representation for ROI {roi} in {path}")
                rdms[roi] = reorder_condensed_rdm(archive[key], neural_ids, common)
            else:
                raise ValueError(f"unrecognized neural key {key!r} in {path}")
    if not rdms:
        raise ValueError(f"{path} contains no patterns__ROI or rdm__ROI arrays")
    return rdms, common


def analyse(
    model_features_path: Path | list[Path] | None,
    neural_dir: Path,
    output_dir: Path,
    *,
    model_rdms_path: Path | list[Path] | None = None,
    rdm_metric: str = "correlation",
    rsa_method: str = "pearson",
    primary_prompt: str = "raw",
    primary_rois: list[str] | None = None,
    primary_locations: list[str] | None = None,
    require_complete: bool = True,
    expected_subjects: int = 8,
) -> dict[str, object]:
    if model_features_path is None:
        model_feature_paths: list[Path] = []
    else:
        model_feature_paths = (
            [model_features_path]
            if isinstance(model_features_path, Path)
            else list(model_features_path)
        )
    if model_rdms_path is None:
        model_rdm_paths: list[Path] = []
    else:
        model_rdm_paths = (
            [model_rdms_path] if isinstance(model_rdms_path, Path) else list(model_rdms_path)
        )
    if not model_feature_paths and not model_rdm_paths:
        raise ValueError("at least one model feature or model RDM archive is required")

    features: dict[str, np.ndarray] = {}
    precomputed_rdms: dict[str, np.ndarray] = {}
    model_ids: np.ndarray | None = None
    if model_feature_paths:
        model_ids, features = load_and_merge_model_features(model_feature_paths)
    if model_rdm_paths:
        rdm_ids, loaded_rdms = load_and_merge_model_rdms(model_rdm_paths)
        if model_ids is None:
            model_ids = rdm_ids
            precomputed_rdms = loaded_rdms
        else:
            if set(rdm_ids.tolist()) != set(model_ids.tolist()):
                raise ValueError("model feature and model RDM archives have different stimulus IDs")
            precomputed_rdms = {
                key: reorder_condensed_rdm(rdm, rdm_ids, model_ids)
                for key, rdm in loaded_rdms.items()
            }
    assert model_ids is not None
    duplicate_keys = sorted(set(features) & set(precomputed_rdms))
    if duplicate_keys:
        raise ValueError(f"duplicate representations across features and RDMs: {duplicate_keys}")
    neural_paths = sorted(neural_dir.glob("*.npz"))
    if not neural_paths:
        raise ValueError(f"no participant .npz files found in {neural_dir}")

    representation_keys = [*features, *precomputed_rdms]
    parsed = {key: parse_representation_key(key) for key in representation_keys}
    score_rows: list[dict[str, object]] = []
    canonical_neural: dict[str, dict[str, np.ndarray]] = {}
    neural_id_sets: dict[str, np.ndarray] = {}

    for neural_path in neural_paths:
        subject = neural_path.stem
        neural_rdms, common_ids = load_neural_file(
            neural_path, model_ids, require_complete=require_complete
        )
        canonical_neural[subject] = neural_rdms
        neural_id_sets[subject] = common_ids
        model_position = {item: i for i, item in enumerate(model_ids.tolist())}
        model_order = np.asarray([model_position[item] for item in common_ids.tolist()], dtype=int)
        model_rdms = {
            key: compute_rdm(values[model_order], metric=rdm_metric)
            for key, values in features.items()
        }
        model_rdms.update(
            {
                key: reorder_condensed_rdm(rdm, model_ids, common_ids)
                for key, rdm in precomputed_rdms.items()
            }
        )
        for roi, neural_rdm in neural_rdms.items():
            neural_rdm = validate_rdm(neural_rdm, common_ids.size)
            for key, model_rdm in model_rdms.items():
                kind, prompt, location = parsed[key]
                score_rows.append(
                    {
                        "subject": subject,
                        "roi": roi,
                        "representation": kind,
                        "prompt": prompt,
                        "location": location,
                        "feature_key": key,
                        "n_images": int(common_ids.size),
                        "rsa": correlate_rdms(model_rdm, neural_rdm, method=rsa_method),
                    }
                )

    scores = pd.DataFrame(score_rows)
    scores["noise_ceiling"] = np.nan
    identical_ids = all(np.array_equal(ids, model_ids) for ids in neural_id_sets.values())
    common_rois = set.intersection(*(set(rdms) for rdms in canonical_neural.values()))
    noise_ceilings: dict[str, dict[str, float]] = {}
    if identical_ids and len(canonical_neural) >= 3:
        for roi in sorted(common_rois):
            subject_rdms = {
                subject: canonical_neural[subject][roi] for subject in canonical_neural
            }
            noise_ceilings[roi] = leave_one_subject_out_noise_ceilings(
                subject_rdms, method=rsa_method
            )
            for subject, ceiling in noise_ceilings[roi].items():
                selector = (scores["subject"] == subject) & (scores["roi"] == roi)
                scores.loc[selector, "noise_ceiling"] = ceiling
    scores["rsa_noise_ceiling_corrected"] = np.where(
        scores["noise_ceiling"] > 0,
        scores["rsa"] / scores["noise_ceiling"],
        np.nan,
    )

    raw = scores[scores["representation"] == "raw"].rename(
        columns={"rsa": "rsa_raw", "feature_key": "raw_key"}
    )
    jlens = scores[scores["representation"] == "jlens"].rename(
        columns={"rsa": "rsa_jlens", "feature_key": "jlens_key"}
    )
    join = ["subject", "roi", "prompt", "location", "n_images"]
    paired = raw[join + ["raw_key", "rsa_raw"]].merge(
        jlens[join + ["jlens_key", "rsa_jlens"]], on=join, how="inner", validate="one_to_one"
    )
    paired["delta_jlens_minus_raw"] = paired["rsa_jlens"] - paired["rsa_raw"]
    primary_roi_set = set(primary_rois or paired["roi"].unique())
    primary_location_set = set(primary_locations or paired["location"].unique())
    primary_selector = (
        (paired["prompt"] == primary_prompt)
        & paired["roi"].isin(primary_roi_set)
        & paired["location"].isin(primary_location_set)
    )
    if not primary_selector.any():
        raise ValueError(
            "the requested primary prompt/ROI/location selection contains no paired contrasts"
        )
    paired["family"] = np.where(
        primary_selector,
        "confirmatory_component",
        "exploratory:" + paired["prompt"].astype(str),
    )

    primary_subject = (
        paired.loc[primary_selector]
        .groupby("subject", as_index=False)
        .agg(
            n_components=("delta_jlens_minus_raw", "size"),
            mean_rsa_raw=("rsa_raw", "mean"),
            mean_rsa_jlens=("rsa_jlens", "mean"),
            mean_delta=("delta_jlens_minus_raw", "mean"),
        )
    )
    expected_components = len(primary_roi_set) * len(primary_location_set)
    observed_subjects = int(scores["subject"].nunique())
    if require_complete:
        if observed_subjects != expected_subjects:
            raise ValueError(
                f"confirmatory analysis requires {expected_subjects} participants; "
                f"found {observed_subjects}"
            )
        if primary_subject.shape[0] != expected_subjects:
            raise ValueError(
                "not every participant has a complete primary raw/J-lens contrast"
            )
        incomplete = primary_subject[
            primary_subject["n_components"] != expected_components
        ]
        if not incomplete.empty:
            details = incomplete[["subject", "n_components"]].to_dict("records")
            raise ValueError(
                f"primary analysis expects {expected_components} ROI x layer components "
                f"per participant; incomplete={details}"
            )
    primary_values = primary_subject["mean_delta"].to_numpy(dtype=float)
    primary_t, primary_p_t = one_sample_t(primary_values)
    primary_sd = (
        float(np.std(primary_values, ddof=1)) if primary_values.size > 1 else float("nan")
    )
    primary_result = {
        "test": "mean J-lens-minus-raw RSA across preregistered ROI x layer components",
        "prompt": primary_prompt,
        "rois": sorted(primary_roi_set),
        "locations": sorted(primary_location_set),
        "n_subjects": int(primary_values.size),
        "n_components_per_subject": sorted(
            primary_subject["n_components"].astype(int).unique().tolist()
        ),
        "mean_rsa_raw": float(primary_subject["mean_rsa_raw"].mean()),
        "mean_rsa_jlens": float(primary_subject["mean_rsa_jlens"].mean()),
        "mean_delta": float(np.mean(primary_values)),
        "sem_delta": float(primary_sd / np.sqrt(primary_values.size)),
        "cohens_dz": (
            float(np.mean(primary_values) / primary_sd) if primary_sd > 0 else float("nan")
        ),
        "t": primary_t,
        "p_t_two_sided": primary_p_t,
        "p_signflip_two_sided": exact_sign_flip_test(primary_values, alternative="two-sided"),
    }

    group_rows: list[dict[str, object]] = []
    grouping = ["family", "roi", "prompt", "location"]
    for keys, frame in paired.groupby(grouping, sort=True):
        values = frame["delta_jlens_minus_raw"].to_numpy(dtype=float)
        t_value, p_t = one_sample_t(values)
        standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        group_rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "n_subjects": int(values.size),
                "mean_rsa_raw": float(frame["rsa_raw"].mean()),
                "mean_rsa_jlens": float(frame["rsa_jlens"].mean()),
                "mean_delta": float(np.mean(values)),
                "sem_delta": float(standard_deviation / np.sqrt(values.size)),
                "cohens_dz": (
                    float(np.mean(values) / standard_deviation)
                    if standard_deviation > 0
                    else float("nan")
                ),
                "t": t_value,
                "p_t_two_sided": p_t,
                "p_signflip_two_sided": exact_sign_flip_test(values, alternative="two-sided"),
            }
        )
    group = pd.DataFrame(group_rows)
    group["p_signflip_fdr"] = np.nan
    for _family, indices in group.groupby("family").groups.items():
        group.loc[indices, "p_signflip_fdr"] = benjamini_hochberg(
            group.loc[indices, "p_signflip_two_sided"].to_numpy()
        )
    group["reject_fdr_0_05"] = group["p_signflip_fdr"] < 0.05

    representation_summary = (
        scores.groupby(["roi", "representation", "prompt", "location"], as_index=False)
        .agg(n_subjects=("subject", "nunique"), mean_rsa=("rsa", "mean"), sem_rsa=("rsa", "sem"))
        .sort_values(["roi", "prompt", "location", "representation"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "rsa_scores.csv", index=False)
    paired.to_csv(output_dir / "paired_contrasts.csv", index=False)
    primary_subject.to_csv(output_dir / "primary_subject_contrasts.csv", index=False)
    group.to_csv(output_dir / "group_contrasts.csv", index=False)
    representation_summary.to_csv(output_dir / "representation_summary.csv", index=False)
    with (output_dir / "primary_test.json").open("w", encoding="utf-8") as handle:
        json.dump(primary_result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_features": [str(path) for path in model_feature_paths],
        "model_rdms": [str(path) for path in model_rdm_paths],
        "neural_dir": str(neural_dir),
        "n_model_images": int(model_ids.size),
        "n_subjects": int(scores["subject"].nunique()),
        "subjects": sorted(scores["subject"].unique().tolist()),
        "rois": sorted(scores["roi"].unique().tolist()),
        "representations": sorted(representation_keys),
        "rdm_metric": rdm_metric,
        "rsa_method": rsa_method,
        "primary_prompt": primary_prompt,
        "primary_rois": sorted(primary_roi_set),
        "primary_locations": sorted(primary_location_set),
        "require_complete": require_complete,
        "expected_subjects": expected_subjects,
        "noise_ceiling_available": bool(noise_ceilings),
        "inference_unit": "participant",
        "primary_test": "exact two-sided paired sign-flip test",
        "multiplicity": "Benjamini-Hochberg within confirmatory/exploratory prompt family",
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-features",
        type=Path,
        action="append",
        help="feature NPZ; repeat to include an aligned benchmark such as MPNet",
    )
    parser.add_argument(
        "--model-rdms",
        type=Path,
        action="append",
        help="precomputed RDM NPZ with rdm__KIND__PROMPT__LOCATION arrays",
    )
    parser.add_argument("--neural-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rdm-metric", default="correlation", choices=["correlation", "cosine"])
    parser.add_argument("--rsa-method", default="pearson", choices=["pearson", "spearman"])
    parser.add_argument("--primary-prompt", default="raw")
    parser.add_argument("--primary-roi", action="append", dest="primary_rois")
    parser.add_argument("--primary-location", action="append", dest="primary_locations")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="allow participant files that omit model stimuli (not the preregistered default)",
    )
    parser.add_argument("--expected-subjects", type=int, default=8)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = analyse(
        args.model_features,
        args.neural_dir,
        args.output_dir,
        model_rdms_path=args.model_rdms,
        rdm_metric=args.rdm_metric,
        rsa_method=args.rsa_method,
        primary_prompt=args.primary_prompt,
        primary_rois=args.primary_rois,
        primary_locations=args.primary_locations,
        require_complete=not args.allow_subset,
        expected_subjects=args.expected_subjects,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
