"""Convert a trusted visuo_llm special515 neural-RDM pickle into safe NPZ files."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .rsa import validate_rdm


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "unnamed_roi"


def convert(pickle_path: Path, index_path: Path, output_dir: Path, *, trusted: bool) -> dict:
    if not trusted:
        raise ValueError(
            "refusing to load pickle without --trusted; only load a file "
            "obtained from a trusted collaborator"
        )
    index = pd.read_csv(index_path)
    nsd_ids = index["nsd_id"].to_numpy(dtype=np.int64)
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - explicitly gated by --trusted
    if not isinstance(payload, dict) or not payload:
        raise ValueError("expected a non-empty dict[subject][roi] pickle")

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_map: dict[str, str] = {}
    subjects = []
    for subject, roi_payload in sorted(payload.items()):
        if not isinstance(roi_payload, dict) or not roi_payload:
            raise ValueError(f"{subject} does not contain a non-empty ROI dictionary")
        arrays: dict[str, np.ndarray] = {"nsd_ids": nsd_ids}
        for roi, rdm in sorted(roi_payload.items()):
            clean = safe_name(str(roi))
            if clean in roi_map and roi_map[clean] != str(roi):
                raise ValueError(
                    f"ROI names collide after sanitization: {roi_map[clean]!r}, {roi!r}"
                )
            roi_map[clean] = str(roi)
            arrays[f"rdm__{clean}"] = validate_rdm(rdm, nsd_ids.size).astype(np.float32)
        subject_name = safe_name(str(subject))
        np.savez_compressed(output_dir / f"{subject_name}.npz", **arrays)
        subjects.append(subject_name)

    manifest = {
        "schema_version": 1,
        "source_pickle": str(pickle_path),
        "index": str(index_path),
        "n_images": int(nsd_ids.size),
        "subjects": subjects,
        "roi_name_map": roi_map,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trusted",
        action="store_true",
        help="confirm that the pickle came from a trusted collaborator",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = convert(args.pickle, args.index, args.output_dir, trusted=args.trusted)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
