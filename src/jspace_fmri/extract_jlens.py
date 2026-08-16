"""Extract matched raw and Jacobian-transported caption representations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np


def load_caption_records(path: Path, limit_images: int | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if limit_images is not None:
        records = records[:limit_images]
    if not records:
        raise ValueError(f"no caption records found in {path}")
    for record in records:
        if not record.get("captions"):
            raise ValueError(f"record for NSD {record.get('nsd_id')} has no captions")
    return records


def flatten_captions(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        for caption_index, caption in enumerate(record["captions"]):
            rows.append(
                {
                    "nsd_id": int(record["nsd_id"]),
                    "coco_id": int(record["coco_id"]),
                    "caption_index": caption_index,
                    "caption": str(caption),
                }
            )
    return rows


def final_content_positions(input_ids, attention_mask, special_ids: set[int]):
    """Return the final attended, non-terminal-special token for each row."""
    import torch

    positions = []
    for row in range(input_ids.shape[0]):
        attended = torch.nonzero(attention_mask[row], as_tuple=False).flatten().tolist()
        if not attended:
            raise ValueError("tokenizer produced an empty sequence")
        position = attended[-1]
        while len(attended) > 1 and int(input_ids[row, position]) in special_ids:
            attended.pop()
            position = attended[-1]
        positions.append(position)
    return torch.as_tensor(positions, device=input_ids.device, dtype=torch.long)


def aggregate_by_image(
    caption_rows: list[dict], caption_features: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    nsd_ids = np.asarray([row["nsd_id"] for row in caption_rows], dtype=np.int64)
    coco_ids = np.asarray([row["coco_id"] for row in caption_rows], dtype=np.int64)
    unique_ids = np.asarray(list(dict.fromkeys(nsd_ids.tolist())), dtype=np.int64)
    image_coco_ids = np.empty(unique_ids.size, dtype=np.int64)
    aggregated = {
        key: np.empty((unique_ids.size, values.shape[1]), dtype=np.float32)
        for key, values in caption_features.items()
    }
    for index, nsd_id in enumerate(unique_ids):
        selector = nsd_ids == nsd_id
        image_coco_ids[index] = coco_ids[selector][0]
        for key, values in caption_features.items():
            aggregated[key][index] = values[selector].mean(axis=0)
    return unique_ids, image_coco_ids, aggregated


def extract(
    config_path: Path,
    captions_path: Path,
    output_dir: Path,
    *,
    limit_images: int | None = None,
    selected_prompts: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    try:
        import torch
        from jlens import ActivationRecorder, JacobianLens, from_hf
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install model dependencies with: uv pip install -e '.[model]'"
        ) from error

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    records = load_caption_records(captions_path, limit_images=limit_images)
    caption_rows = flatten_captions(records)
    prompt_templates = config["prompt_templates"]
    if selected_prompts is None:
        selected_prompts = list(prompt_templates)
    unknown_prompts = sorted(set(selected_prompts) - set(prompt_templates))
    if unknown_prompts:
        raise ValueError(f"unknown prompt conditions: {unknown_prompts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "caption_features.npz",
        output_dir / "image_features.npz",
        output_dir / "extraction_manifest.json",
    ]
    if not overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"outputs already exist in {output_dir}; pass --overwrite to replace")

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = config.get("device", "auto")
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else requested_device
    )
    if device == "auto":
        device = "cpu"
    dtype_name = config.get("dtype", "bfloat16")
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    dtype = dtype_by_name[dtype_name]
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU execution is unsupported; use float32")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config.get("model_revision")
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], revision=config.get("model_revision"), dtype=dtype
    )
    model.to(device)
    model.eval()
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
    lens = JacobianLens.from_pretrained(
        config["lens_repo"],
        filename=config["lens_filename"],
        revision=config.get("lens_revision"),
    )
    if lens.d_model != lens_model.d_model:
        raise ValueError(
            f"lens width {lens.d_model} does not match model width {lens_model.d_model}"
        )
    layers = [int(layer) for layer in config["layers"]]
    unavailable = sorted(set(layers) - set(lens.source_layers))
    if unavailable:
        raise ValueError(f"requested layers absent from fitted lens: {unavailable}")
    if any(layer < 0 or layer >= lens_model.n_layers for layer in layers):
        raise ValueError(f"layers {layers} invalid for {lens_model.n_layers}-layer model")

    # Keep only preregistered Jacobians and place them once, instead of copying a
    # d_model × d_model matrix from CPU on every caption batch.
    lens.jacobians = {layer: lens.jacobians[layer].to(device) for layer in layers}
    lens.source_layers = sorted(lens.jacobians)
    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    batch_size = int(config.get("batch_size", 8))
    max_length = int(config.get("max_length", 128))
    special_ids = set(int(item) for item in tokenizer.all_special_ids)
    caption_features: dict[str, np.ndarray] = {}

    def apply_final_norm(residual):
        norm = lens_model._final_norm  # pinned jlens adapter; recorded in manifest
        norm_parameter = next(norm.parameters(), None)
        target_dtype = norm_parameter.dtype if norm_parameter is not None else residual.dtype
        return norm(residual.to(dtype=target_dtype)).float()

    with torch.inference_mode():
        for prompt_name in selected_prompts:
            template = prompt_templates[prompt_name]
            outputs: dict[str, list[np.ndarray]] = {}
            for start in range(0, len(caption_rows), batch_size):
                rows = caption_rows[start : start + batch_size]
                texts = [template.format(caption=row["caption"]) for row in rows]
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                positions = final_content_positions(input_ids, attention_mask, special_ids)
                with ActivationRecorder(lens_model.layers, at=record_at) as recorder:
                    lens_model._text_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                for layer in layers:
                    residual = recorder.activations[layer][row_indices, positions].float()
                    raw = apply_final_norm(residual)
                    transported = apply_final_norm(lens.transport(residual, layer))
                    outputs.setdefault(f"raw__{prompt_name}__layer_{layer:02d}", []).append(
                        raw.cpu().numpy()
                    )
                    outputs.setdefault(f"jlens__{prompt_name}__layer_{layer:02d}", []).append(
                        transported.cpu().numpy()
                    )
                final_residual = recorder.activations[final_layer][row_indices, positions]
                outputs.setdefault(f"final__{prompt_name}__layer_final", []).append(
                    apply_final_norm(final_residual).cpu().numpy()
                )
                del recorder, input_ids, attention_mask
            for key, chunks in outputs.items():
                caption_features[key] = np.concatenate(chunks).astype(np.float32, copy=False)

    repeated_nsd_ids = np.asarray([row["nsd_id"] for row in caption_rows], dtype=np.int64)
    repeated_coco_ids = np.asarray([row["coco_id"] for row in caption_rows], dtype=np.int64)
    caption_indices = np.asarray([row["caption_index"] for row in caption_rows], dtype=np.int16)
    np.savez_compressed(
        output_dir / "caption_features.npz",
        nsd_ids=repeated_nsd_ids,
        coco_ids=repeated_coco_ids,
        caption_indices=caption_indices,
        **caption_features,
    )
    image_nsd_ids, image_coco_ids, image_features = aggregate_by_image(
        caption_rows, caption_features
    )
    np.savez_compressed(
        output_dir / "image_features.npz",
        nsd_ids=image_nsd_ids,
        coco_ids=image_coco_ids,
        **image_features,
    )

    try:
        jlens_version = importlib.metadata.version("jlens")
    except importlib.metadata.PackageNotFoundError:
        jlens_version = "unknown"
    manifest = {
        "schema_version": 1,
        "config": config,
        "config_path": str(config_path),
        "captions_path": str(captions_path),
        "n_images": int(image_nsd_ids.size),
        "n_captions": len(caption_rows),
        "selected_prompts": selected_prompts,
        "representations": sorted(image_features),
        "device_used": str(device),
        "dtype_used": dtype_name,
        "model_layers": lens_model.n_layers,
        "d_model": lens_model.d_model,
        "lens_n_prompts": lens.n_prompts,
        "jlens_version": jlens_version,
        "token_readout": "last attended token after removing terminal special tokens",
        "caption_pooling": "arithmetic mean within image",
        "generation_performed": False,
        "terminology": (
            "J-lens-derived/Jacobian-transported residual, "
            "not sparse J-space projection"
        ),
    }
    with (output_dir / "extraction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = extract(
        args.config,
        args.captions,
        args.output_dir,
        limit_images=args.limit_images,
        selected_prompts=args.prompts,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
