"""Validate and merge single-condition recovery RDM archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import RESERVED_MODEL_KEYS, parse_representation_key
from .rsa import reorder_condensed_rdm, validate_ids, validate_rdm


def merge_recovery_archives(paths: list[Path], output_path: Path) -> dict[str, object]:
    """Merge aligned RDM archives, adding the canonical ``rdm__`` key prefix."""
    if not paths:
        raise ValueError("at least one input archive is required")

    canonical_ids: np.ndarray | None = None
    canonical_coco_ids: np.ndarray | None = None
    merged: dict[str, np.ndarray] = {}
    source_manifests: list[dict[str, object]] = []

    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "nsd_ids" not in archive:
                raise ValueError(f"{path} has no nsd_ids")
            ids = validate_ids(archive["nsd_ids"], name=f"{path.name} nsd_ids")
            coco_ids = np.asarray(archive["coco_ids"]) if "coco_ids" in archive else None
            if canonical_ids is None:
                canonical_ids = ids
                canonical_coco_ids = coco_ids
            elif set(ids.tolist()) != set(canonical_ids.tolist()):
                raise ValueError(f"{path} has a different NSD stimulus set")

            if coco_ids is not None and canonical_coco_ids is not None:
                source_position = {item: i for i, item in enumerate(ids.tolist())}
                order = np.asarray([source_position[item] for item in canonical_ids.tolist()])
                if not np.array_equal(coco_ids[order], canonical_coco_ids):
                    raise ValueError(f"{path} has inconsistent COCO IDs")

            if "manifest_json" in archive:
                source_manifests.append(json.loads(str(archive["manifest_json"])))

            for archive_key in archive.files:
                if archive_key in RESERVED_MODEL_KEYS:
                    continue
                representation_key = archive_key.removeprefix("rdm__")
                parse_representation_key(representation_key)
                output_key = f"rdm__{representation_key}"
                if output_key in merged:
                    raise ValueError(f"duplicate representation {output_key!r}")
                rdm = validate_rdm(archive[archive_key], ids.size)
                if not np.array_equal(ids, canonical_ids):
                    rdm = reorder_condensed_rdm(rdm, ids, canonical_ids)
                merged[output_key] = rdm.astype(np.float32)

    assert canonical_ids is not None
    payload: dict[str, np.ndarray] = {"nsd_ids": canonical_ids}
    if canonical_coco_ids is not None:
        payload["coco_ids"] = canonical_coco_ids
    manifest = {
        "schema_version": 1,
        "event": "JSPACE_RECOVERY_RDM_MERGE",
        "inputs": [str(path) for path in paths],
        "n_images": int(canonical_ids.size),
        "representations": sorted(key.removeprefix("rdm__") for key in merged),
        "source_manifests": source_manifests,
    }
    payload["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    payload.update(merged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_recovery_archives(args.input, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
