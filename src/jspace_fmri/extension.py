"""Apply the frozen exploratory-extension analysis to paired RSA scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_model_rdms, parse_representation_key
from .rsa import benjamini_hochberg, correlate_rdms, exact_sign_flip_test, one_sample_t

NEW_CONDITIONS = {
    ("raw", "layer_09"),
    ("raw", "layer_34"),
    ("imagine", "layer_09"),
    ("imagine", "layer_18"),
    ("imagine", "layer_27"),
    ("imagine", "layer_34"),
}


def _summarise(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    t_value, p_t = one_sample_t(array)
    standard_deviation = float(np.std(array, ddof=1))
    return {
        "n_subjects": int(array.size),
        "mean_delta": float(np.mean(array)),
        "sem_delta": float(standard_deviation / np.sqrt(array.size)),
        "cohens_dz": float(np.mean(array) / standard_deviation)
        if standard_deviation > 0
        else float("nan"),
        "t": t_value,
        "p_t_two_sided": p_t,
        "p_signflip_two_sided": exact_sign_flip_test(array),
    }


def _group_summary(
    frame: pd.DataFrame, grouping: list[str], value: str
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(grouping, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(grouping, key_tuple, strict=True)), **_summarise(group[value])})
    return pd.DataFrame(rows)


def analyse_extension(
    paired_path: Path, model_rdms_path: Path, output_dir: Path
) -> dict[str, object]:
    paired = pd.read_csv(paired_path)
    required = {
        "subject",
        "roi",
        "prompt",
        "location",
        "delta_jlens_minus_raw",
    }
    missing = sorted(required - set(paired.columns))
    if missing:
        raise ValueError(f"paired table is missing columns: {missing}")

    condition = list(zip(paired["prompt"], paired["location"], strict=True))
    extension = paired[[item in NEW_CONDITIONS for item in condition]].copy()
    expected_rows = 18 * 8
    if extension.shape[0] != expected_rows:
        raise ValueError(
            f"frozen extension requires {expected_rows} rows (18 cells x 8 participants); "
            f"found {extension.shape[0]}"
        )
    cell_counts = extension.groupby(["roi", "prompt", "location"])["subject"].nunique()
    if cell_counts.size != 18 or not (cell_counts == 8).all():
        raise ValueError("the frozen 18-cell family is incomplete")

    group = _group_summary(
        extension,
        ["roi", "prompt", "location"],
        "delta_jlens_minus_raw",
    )
    group["p_signflip_fdr_18_cells"] = benjamini_hochberg(
        group["p_signflip_two_sided"].to_numpy()
    )
    group["reject_fdr_0_05"] = group["p_signflip_fdr_18_cells"] < 0.05

    final_prompt = paired[paired["prompt"].isin(["raw", "imagine"])].copy()
    trajectory_subject = (
        final_prompt.groupby(["subject", "prompt", "location"], as_index=False)
        .agg(mean_delta=("delta_jlens_minus_raw", "mean"))
        .sort_values(["prompt", "location", "subject"])
    )
    trajectory_group = _group_summary(
        trajectory_subject, ["prompt", "location"], "mean_delta"
    )

    raw = final_prompt[final_prompt["prompt"] == "raw"].rename(
        columns={"delta_jlens_minus_raw": "delta_raw_prompt"}
    )
    imagine = final_prompt[final_prompt["prompt"] == "imagine"].rename(
        columns={"delta_jlens_minus_raw": "delta_imagine_prompt"}
    )
    moderation = raw[["subject", "roi", "location", "delta_raw_prompt"]].merge(
        imagine[["subject", "roi", "location", "delta_imagine_prompt"]],
        on=["subject", "roi", "location"],
        validate="one_to_one",
    )
    moderation["prompt_moderation"] = (
        moderation["delta_imagine_prompt"] - moderation["delta_raw_prompt"]
    )
    moderation_group = _group_summary(
        moderation, ["roi", "location"], "prompt_moderation"
    )

    no_final = paired[paired["prompt"] == "raw_nofinalnorm"].rename(
        columns={"delta_jlens_minus_raw": "delta_no_final_norm"}
    )
    final_norm = paired[paired["prompt"] == "raw"].rename(
        columns={"delta_jlens_minus_raw": "delta_final_norm"}
    )
    norm = final_norm[
        ["subject", "roi", "location", "delta_final_norm"]
    ].merge(
        no_final[["subject", "roi", "location", "delta_no_final_norm"]],
        on=["subject", "roi", "location"],
        validate="one_to_one",
    )
    norm["moderation_no_final_minus_final"] = (
        norm["delta_no_final_norm"] - norm["delta_final_norm"]
    )
    norm_group = _group_summary(
        norm, ["roi", "location"], "moderation_no_final_minus_final"
    )

    _ids, model_rdms = load_model_rdms(model_rdms_path)
    model_rows = []
    parsed = {key: parse_representation_key(key) for key in model_rdms}
    for key, (kind, prompt, location) in parsed.items():
        if kind != "raw":
            continue
        jlens_key = f"jlens__{prompt}__{location}"
        if jlens_key not in model_rdms:
            raise ValueError(f"missing paired model RDM {jlens_key}")
        model_rows.append(
            {
                "prompt": prompt,
                "location": location,
                "raw_jlens_rdm_pearson": correlate_rdms(
                    model_rdms[key], model_rdms[jlens_key], method="pearson"
                ),
            }
        )
    model_correlations = pd.DataFrame(model_rows).sort_values(["prompt", "location"])

    output_dir.mkdir(parents=True, exist_ok=True)
    extension.to_csv(output_dir / "extension_participant_contrasts.csv", index=False)
    group.to_csv(output_dir / "extension_group_contrasts.csv", index=False)
    trajectory_subject.to_csv(output_dir / "layer_trajectory_participants.csv", index=False)
    trajectory_group.to_csv(output_dir / "layer_trajectory_group.csv", index=False)
    moderation.to_csv(output_dir / "prompt_moderation_participants.csv", index=False)
    moderation_group.to_csv(output_dir / "prompt_moderation_group.csv", index=False)
    norm.to_csv(output_dir / "normalization_control_participants.csv", index=False)
    norm_group.to_csv(output_dir / "normalization_control_group.csv", index=False)
    model_correlations.to_csv(output_dir / "model_rdm_correlations.csv", index=False)

    summary: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "exploratory extension frozen after the original raw-layer-18/27 result",
        "n_subjects": int(extension["subject"].nunique()),
        "n_new_cells": int(group.shape[0]),
        "fdr_family": "18 new prompt-by-layer-by-ROI cells",
        "n_fdr_rejections": int(group["reject_fdr_0_05"].sum()),
        "new_cells_negative": int((group["mean_delta"] < 0).sum()),
        "new_cells_positive": int((group["mean_delta"] > 0).sum()),
        "paired_table": str(paired_path),
        "model_rdms": str(model_rdms_path),
        "claim_boundary": (
            "Dense Jacobian transport is an indirect J-lens diagnostic, not formal sparse "
            "non-negative J-space and not evidence of a homologous neural workspace."
        ),
    }
    with (output_dir / "extension_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--model-rdms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyse_extension(args.paired, args.model_rdms, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
