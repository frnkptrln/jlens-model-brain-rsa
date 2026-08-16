"""Prepare the shared-515 NSD caption set from small public metadata files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

NSD_BASE = "https://natural-scenes-dataset.s3.amazonaws.com"
STIM_INFO_URL = f"{NSD_BASE}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
BEHAVIOR_URL = f"{NSD_BASE}/nsddata/ppdata/{{subject}}/behav/responses.tsv"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "jspace-fmri-pilot/0.1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(temporary, destination)
    return destination


def extract_caption_annotations(
    archive: Path, raw_dir: Path, *, force: bool = False
) -> dict[str, Path]:
    targets = {
        "train2017": raw_dir / "captions_train2017.json",
        "val2017": raw_dir / "captions_val2017.json",
    }
    with zipfile.ZipFile(archive) as zipped:
        for split, target in targets.items():
            if target.exists() and not force:
                continue
            member = f"annotations/captions_{split}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
    return targets


def shared_three_repeat_ids(response_paths: list[Path]) -> list[int]:
    shared: set[int] | None = None
    for path in response_paths:
        table = pd.read_csv(path, sep="\t", usecols=["73KID"])
        counts = Counter(int(value) for value in table["73KID"].dropna())
        repeated = {item for item, count in counts.items() if count == 3}
        shared = repeated if shared is None else shared & repeated
    if shared is None:
        raise ValueError("no response files supplied")
    return sorted(shared)


def load_captions(paths: dict[str, Path]) -> dict[str, dict[int, list[str]]]:
    output: dict[str, dict[int, list[str]]] = {}
    for split, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        by_image: defaultdict[int, list[str]] = defaultdict(list)
        for annotation in payload["annotations"]:
            by_image[int(annotation["image_id"])].append(annotation["caption"].strip())
        output[split] = dict(by_image)
    return output


def build_records(
    shared_ids: list[int], stim_info_path: Path, caption_paths: dict[str, Path]
) -> list[dict[str, object]]:
    stim_info = pd.read_csv(stim_info_path)
    stim_info["nsd_id"] = stim_info["nsdId"].astype(int) + 1
    by_nsd = stim_info.set_index("nsd_id", drop=False)
    captions = load_captions(caption_paths)
    records: list[dict[str, object]] = []
    for nsd_id in shared_ids:
        if nsd_id not in by_nsd.index:
            raise ValueError(f"NSD ID {nsd_id} is absent from stimulus metadata")
        row = by_nsd.loc[nsd_id]
        split = str(row["cocoSplit"])
        coco_id = int(row["cocoId"])
        image_captions = captions.get(split, {}).get(coco_id, [])
        if not image_captions:
            raise ValueError(f"no captions found for NSD {nsd_id}, COCO {split}/{coco_id}")
        records.append(
            {
                "nsd_id": nsd_id,
                "nsd_index": int(row["nsdId"]),
                "coco_id": coco_id,
                "coco_split": split,
                "captions": image_captions,
            }
        )
    return records


def prepare(data_dir: Path, *, force: bool = False, expected_count: int = 515) -> dict[str, object]:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    stim_info = download(STIM_INFO_URL, raw_dir / "nsd_stim_info_merged.csv", force=force)
    response_paths = []
    sources: dict[str, str] = {"stim_info": STIM_INFO_URL, "coco_annotations": COCO_ANNOTATIONS_URL}
    for number in range(1, 9):
        subject = f"subj{number:02d}"
        url = BEHAVIOR_URL.format(subject=subject)
        sources[f"responses_{subject}"] = url
        response_paths.append(download(url, raw_dir / f"{subject}_responses.tsv", force=force))

    archive = download(COCO_ANNOTATIONS_URL, raw_dir / "annotations_trainval2017.zip", force=force)
    caption_paths = extract_caption_annotations(archive, raw_dir, force=force)
    shared_ids = shared_three_repeat_ids(response_paths)
    if len(shared_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} shared three-repeat images, found {len(shared_ids)}"
        )
    records = build_records(shared_ids, stim_info, caption_paths)

    jsonl_path = processed_dir / "shared515_captions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    index_path = processed_dir / "shared515_index.csv"
    pd.DataFrame(
        [
            {
                "order": order,
                "nsd_id": record["nsd_id"],
                "nsd_index": record["nsd_index"],
                "coco_id": record["coco_id"],
                "coco_split": record["coco_split"],
                "n_captions": len(record["captions"]),
            }
            for order, record in enumerate(records)
        ]
    ).to_csv(index_path, index=False)

    input_files = [stim_info, *response_paths, archive, *caption_paths.values()]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "n_images": len(records),
        "n_captions": sum(len(record["captions"]) for record in records),
        "caption_count_distribution": dict(
            sorted(Counter(len(record["captions"]) for record in records).items())
        ),
        "sources": sources,
        "sha256": {path.name: sha256(path) for path in input_files},
        "outputs": {"captions": str(jsonl_path), "index": str(index_path)},
    }
    manifest_path = processed_dir / "shared515_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="redownload source files")
    parser.add_argument("--expected-count", type=int, default=515)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    manifest = prepare(args.data_dir, force=args.force, expected_count=args.expected_count)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
