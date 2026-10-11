"""Read-only reconstruction of the published confirmatory inference from CSVs.

This audits released summaries; it does not recompute neural/model RDMs or run
the model. No source or published result is rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from itertools import product
from pathlib import Path

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = tuple(f"subj{number:02d}" for number in range(1, 9))
ROIS = ("lateral", "parietal", "ventral")
LAYERS = ("layer_18", "layer_27")
FILES = (
    "rsa_scores.csv",
    "paired_contrasts.csv",
    "primary_subject_contrasts.csv",
    "primary_test.json",
    "result_summary.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, label: str) -> None:
    if (
        not math.isfinite(actual)
        or not math.isfinite(expected)
        or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError(f"{label}: recomputed {actual!r}, archived {expected!r}")


def unique_index(rows: list[dict], fields: tuple[str, ...], label: str) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"{label}: duplicate identity {key}")
        result[key] = row
    return result


def audit(directory: Path) -> dict:
    primary = json.loads((directory / "primary_test.json").read_text())
    summary = json.loads((directory / "result_summary.json").read_text())
    if (
        primary["prompt"] != "raw"
        or tuple(sorted(primary["rois"])) != ROIS
        or tuple(sorted(primary["locations"])) != LAYERS
        or primary["n_subjects"] != 8
        or primary["n_components_per_subject"] != [6]
    ):
        raise ValueError("published primary selection differs from the frozen 8 × 3 × 2 contract")
    paired = unique_index(
        read_csv(directory / "paired_contrasts.csv"),
        ("subject", "roi", "prompt", "location"),
        "paired contrasts",
    )
    expected = set(product(SUBJECTS, ROIS, ("raw",), LAYERS))
    if set(paired) != expected:
        raise ValueError(
            "paired contrasts: expected exactly 48 cells; "
            f"missing={sorted(expected - set(paired))}, extra={sorted(set(paired) - expected)}"
        )

    scores = unique_index(
        read_csv(directory / "rsa_scores.csv"), ("subject", "roi", "feature_key"), "RSA scores"
    )
    expected_scores = {
        (subject, roi, f"{kind}__raw__{layer}")
        for subject, roi, _, layer in expected
        for kind in ("raw", "jlens")
    }
    if set(scores) != expected_scores:
        raise ValueError(
            "RSA scores must contain exactly the 96 expected participant/ROI/representation cells"
        )
    for (subject, roi, _, layer), row in paired.items():
        if row["family"] != "confirmatory_component" or int(row["n_images"]) != 515:
            raise ValueError(f"{subject}/{roi}/{layer}: unexpected family or image count")
        for kind in ("raw", "jlens"):
            key = f"{kind}__raw__{layer}"
            if row[f"{kind}_key"] != key:
                raise ValueError(f"{subject}/{roi}/{layer}: representation key mismatch")
            score = scores[(subject, roi, key)]
            if (
                score["representation"] != kind
                or score["prompt"] != "raw"
                or score["location"] != layer
            ):
                raise ValueError(f"{subject}/{roi}/{key}: RSA score metadata mismatch")
            if int(score["n_images"]) != 515:
                raise ValueError("RSA score image count differs from 515")
            close(float(row[f"rsa_{kind}"]), float(score["rsa"]), f"{subject}/{roi}/{key}")
        close(
            float(row["rsa_jlens"]) - float(row["rsa_raw"]),
            float(row["delta_jlens_minus_raw"]),
            f"{subject}/{roi}/{layer} delta",
        )

    archived_subjects = unique_index(
        read_csv(directory / "primary_subject_contrasts.csv"), ("subject",), "participant summary"
    )
    if set(archived_subjects) != {(subject,) for subject in SUBJECTS}:
        raise ValueError("participant summary must contain each of the eight subjects exactly once")
    subjects = []
    for subject in SUBJECTS:
        rows = [paired[(subject, roi, "raw", layer)] for roi, layer in product(ROIS, LAYERS)]
        row = {
            "subject": subject,
            "n_components": len(rows),
            "mean_rsa_raw": statistics.mean(float(item["rsa_raw"]) for item in rows),
            "mean_rsa_jlens": statistics.mean(float(item["rsa_jlens"]) for item in rows),
            "mean_delta": statistics.mean(float(item["delta_jlens_minus_raw"]) for item in rows),
        }
        for field in ("n_components", "mean_rsa_raw", "mean_rsa_jlens", "mean_delta"):
            close(row[field], float(archived_subjects[(subject,)][field]), f"{subject}/{field}")
        subjects.append(row)

    deltas = [row["mean_delta"] for row in subjects]
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0:
        raise ValueError("participant contrasts have zero variance; t and dz are undefined")
    sem = sd / math.sqrt(len(deltas))
    # Independent enumeration, after averaging the six cells within each participant.
    permutations = [
        statistics.mean(value * sign for value, sign in zip(deltas, signs, strict=True))
        for signs in product((-1, 1), repeat=len(deltas))
    ]
    extreme = sum(abs(value) >= abs(mean) - 32 * sys.float_info.epsilon for value in permutations)
    recomputed = {
        "n_subjects": len(deltas),
        "mean_delta": mean,
        "sem_delta": sem,
        "mean_rsa_raw": statistics.mean(row["mean_rsa_raw"] for row in subjects),
        "mean_rsa_jlens": statistics.mean(row["mean_rsa_jlens"] for row in subjects),
        "cohens_dz": mean / sd,
        "t": mean / sem,
        "p_t_two_sided": float(2 * student_t.sf(abs(mean / sem), df=len(deltas) - 1)),
        "p_signflip_two_sided": extreme / len(permutations),
    }
    for field, value in recomputed.items():
        close(value, float(primary[field]), f"primary/{field}")
        close(value, float(summary["primary"][field]), f"result_summary/primary/{field}")
    if summary["inference_unit"] != "participant":
        raise ValueError("inference unit must remain participant")
    return {
        "status": "pass",
        "method": "released-cell-to-participant-to-group reconstruction",
        "inference_unit": "participant",
        "cell_count": len(paired),
        "sign_assignments": len(permutations),
        "extreme_assignments": extreme,
        "negative_participants": sum(value < 0 for value in deltas),
        "primary": recomputed,
        "files_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES
        },
        "claim_boundary": (
            "Checks consistency of released summaries; "
            "no model, NSD data, RDM, extension or causal-claim replication."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/confirmatory")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(audit(args.analysis_dir), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"confirmatory audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
