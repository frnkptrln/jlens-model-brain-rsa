"""Extract the Nature-paper all-mpnet-base-v2 caption baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .extract_jlens import aggregate_by_image, flatten_captions, load_caption_records


def extract_mpnet(
    captions_path: Path,
    output_dir: Path,
    *,
    model_name: str = "all-mpnet-base-v2",
    revision: str = "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    batch_size: int = 64,
    limit_images: int | None = None,
    overwrite: bool = False,
) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "install MPNet dependencies with: uv pip install -e '.[mpnet]'"
        ) from error

    records = load_caption_records(captions_path, limit_images=limit_images)
    rows = flatten_captions(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "image_features.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace")

    model = SentenceTransformer(model_name, revision=revision)
    captions = [row["caption"] for row in rows]
    features = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    key = "mpnet__raw__sentence"
    repeated_nsd_ids = np.asarray([row["nsd_id"] for row in rows], dtype=np.int64)
    repeated_coco_ids = np.asarray([row["coco_id"] for row in rows], dtype=np.int64)
    caption_indices = np.asarray([row["caption_index"] for row in rows], dtype=np.int16)
    np.savez_compressed(
        output_dir / "caption_features.npz",
        nsd_ids=repeated_nsd_ids,
        coco_ids=repeated_coco_ids,
        caption_indices=caption_indices,
        **{key: features},
    )
    image_nsd_ids, image_coco_ids, aggregated = aggregate_by_image(rows, {key: features})
    np.savez_compressed(
        output_path,
        nsd_ids=image_nsd_ids,
        coco_ids=image_coco_ids,
        **aggregated,
    )
    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "model_revision": revision,
        "captions_path": str(captions_path),
        "n_images": int(image_nsd_ids.size),
        "n_captions": len(rows),
        "caption_pooling": "arithmetic mean within image",
        "normalize_embeddings": False,
    }
    with (output_dir / "extraction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="all-mpnet-base-v2")
    parser.add_argument(
        "--revision", default="e8c3b32edf5434bc2275fc9bab85f82640a19130"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = extract_mpnet(
        args.captions,
        args.output_dir,
        model_name=args.model_name,
        revision=args.revision,
        batch_size=args.batch_size,
        limit_images=args.limit_images,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
