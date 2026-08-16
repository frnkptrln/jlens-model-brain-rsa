"""Generate a labelled synthetic dataset to validate the RSA implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def simulate(output_dir: Path, *, n_images: int = 48, n_subjects: int = 8, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    neural_dir = output_dir / "neural"
    neural_dir.mkdir(exist_ok=True)

    nsd_ids = np.arange(1, n_images + 1, dtype=np.int64)
    latent = rng.normal(size=(n_images, 12))
    raw = latent + rng.normal(scale=1.25, size=latent.shape)
    jlens = latent + rng.normal(scale=0.25, size=latent.shape)
    exploratory_raw = latent + rng.normal(scale=1.0, size=latent.shape)
    exploratory_jlens = latent + rng.normal(scale=0.45, size=latent.shape)
    np.savez_compressed(
        output_dir / "model_features.npz",
        nsd_ids=nsd_ids,
        raw__raw__layer_02=raw.astype(np.float32),
        jlens__raw__layer_02=jlens.astype(np.float32),
        raw__imagine__layer_02=exploratory_raw.astype(np.float32),
        jlens__imagine__layer_02=exploratory_jlens.astype(np.float32),
    )

    high_projection = rng.normal(size=(latent.shape[1], 36))
    for subject_index in range(1, n_subjects + 1):
        high_level = latent @ high_projection + rng.normal(scale=1.5, size=(n_images, 36))
        # Independent null patterns ensure there is no injected cross-subject
        # early-visual relationship with either language representation.
        early_visual = rng.normal(size=(n_images, 28))
        np.savez_compressed(
            neural_dir / f"subj{subject_index:02d}.npz",
            nsd_ids=nsd_ids,
            patterns__high_level=high_level.astype(np.float32),
            patterns__early_visual=early_visual.astype(np.float32),
        )

    manifest = {
        "synthetic": True,
        "warning": "For software validation only; never report as empirical evidence.",
        "seed": seed,
        "n_images": n_images,
        "n_subjects": n_subjects,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=48)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = simulate(
        args.output_dir, n_images=args.n_images, n_subjects=args.n_subjects, seed=args.seed
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
