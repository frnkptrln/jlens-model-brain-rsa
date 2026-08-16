"""Recompute shared-515 NSD stream-ROI RDMs from public fsaverage betas.

The implementation follows the reference ``visuo_llm`` preprocessing while
avoiding its all-session, all-vertex memory spike.  Vertices are independent
during session-wise z-scoring, so selecting the requested ROIs before that
operation is mathematically equivalent to selecting them afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import zscore

from .nsd import NSD_BASE, download, sha256

BETAS_URL = (
    f"{NSD_BASE}/nsddata_betas/ppdata/{{subject}}/fsaverage/"
    "betas_fithrf_GLMdenoise_RR/{hemi}.betas_session{session:02d}.mgh"
)
STREAM_MASK_URL = f"{NSD_BASE}/nsddata/freesurfer/fsaverage/label/{{hemi}}.streams.mgz"
STREAM_CTAB_URL = f"{NSD_BASE}/nsddata/freesurfer/fsaverage/label/streams.mgz.ctab"
BEHAVIOR_URL = f"{NSD_BASE}/nsddata/ppdata/{{subject}}/behav/responses.tsv"

DEFAULT_ROIS = {5: "ventral", 6: "lateral", 7: "parietal"}
VISUO_LLM_COMMIT = "a60e0eafb8d02841159e344adb732062734bc302"


@dataclass(frozen=True)
class TargetTrial:
    trial: int
    target: int
    repeat: int


def read_nsd_ids(index_path: Path) -> np.ndarray:
    table = pd.read_csv(index_path, usecols=["nsd_id"])
    ids = table["nsd_id"].to_numpy(dtype=np.int64)
    if ids.size < 3 or np.unique(ids).size != ids.size:
        raise ValueError("index must contain at least three unique nsd_id values")
    return ids


def build_schedule(
    behavior: pd.DataFrame, target_ids: np.ndarray, *, repeats: int = 3
) -> dict[int, list[TargetTrial]]:
    """Map session-local trials to target and repeat slots."""
    required = {"SESSION", "73KID"}
    missing = required - set(behavior.columns)
    if missing:
        raise ValueError(f"behavior table is missing columns: {sorted(missing)}")
    target_ids = np.asarray(target_ids, dtype=np.int64)
    target_to_position = {int(item): index for index, item in enumerate(target_ids)}
    seen = np.zeros(target_ids.size, dtype=np.int16)
    schedule: dict[int, list[TargetTrial]] = {}
    for session in sorted(int(value) for value in behavior["SESSION"].dropna().unique()):
        session_rows = behavior.loc[behavior["SESSION"] == session].reset_index(drop=True)
        selected: list[TargetTrial] = []
        for trial, value in enumerate(session_rows["73KID"]):
            if pd.isna(value):
                continue
            target = target_to_position.get(int(value))
            if target is None:
                continue
            repeat = int(seen[target])
            if repeat >= repeats:
                raise ValueError(f"NSD ID {int(value)} occurs more than {repeats} times")
            selected.append(TargetTrial(trial=trial, target=target, repeat=repeat))
            seen[target] += 1
        if selected:
            schedule[session] = selected
    bad = np.flatnonzero(seen != repeats)
    if bad.size:
        examples = [(int(target_ids[index]), int(seen[index])) for index in bad[:5]]
        raise ValueError(
            f"{bad.size} target images do not have exactly {repeats} repeats; first={examples}"
        )
    return schedule


def parse_ctab(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                labels[int(fields[0])] = fields[1]
    return labels


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(nib.load(path).get_fdata()).squeeze()
    if mask.ndim != 1:
        raise ValueError(f"stream mask must be one-dimensional, got {mask.shape}")
    return mask.astype(np.int8, copy=False)


def zscore_selected_vertices(beta_path: Path, vertex_indices: np.ndarray) -> np.ndarray:
    """Load and z-score selected vertices exactly as the reference code does."""
    image = nib.load(beta_path, mmap=True)
    full = np.asanyarray(image.dataobj).squeeze()
    if full.ndim != 2:
        raise ValueError(f"beta file must reduce to [vertices, trials], got {full.shape}")
    if vertex_indices.size and int(vertex_indices[-1]) >= full.shape[0]:
        raise ValueError("ROI mask has more vertices than the beta file")
    # get_fdata() in the reference code promotes MGH values to float64 before
    # scipy.stats.zscore. Fancy-indexing the mmap first avoids loading unrelated
    # vertices while preserving the same row-wise calculation.
    selected = np.asarray(full[vertex_indices, :], dtype=np.float64)
    standardized = zscore(selected, axis=-1)
    return standardized.astype(np.float32, copy=False)


def _download_with_retries(url: str, destination: Path, attempts: int = 3) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    error: Exception | None = None
    for attempt in range(attempts):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "jspace-fmri-pilot/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            os.replace(temporary, destination)
            return destination
        except Exception as caught:  # pragma: no cover - exercised only on network faults
            error = caught
            temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url} after {attempts} attempts") from error


def prepare_metadata(metadata_dir: Path, subjects: list[str]) -> dict[str, Path]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for hemi in ("lh", "rh"):
        paths[f"mask_{hemi}"] = download(
            STREAM_MASK_URL.format(hemi=hemi), metadata_dir / f"{hemi}.streams.mgz"
        )
    paths["ctab"] = download(STREAM_CTAB_URL, metadata_dir / "streams.mgz.ctab")
    for subject in subjects:
        paths[f"behavior_{subject}"] = download(
            BEHAVIOR_URL.format(subject=subject), metadata_dir / f"{subject}_responses.tsv"
        )
    return paths


def _allocate_repeats(
    n_images: int,
    masks: dict[str, np.ndarray],
    roi_ids: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[tuple[str, int], slice]]:
    offsets: dict[tuple[str, int], slice] = {}
    arrays: dict[int, np.ndarray] = {}
    for roi_id in roi_ids:
        cursor = 0
        for hemi in ("lh", "rh"):
            count = int(np.sum(masks[hemi] == roi_id))
            offsets[(hemi, roi_id)] = slice(cursor, cursor + count)
            cursor += count
        arrays[roi_id] = np.full((3, n_images, cursor), np.nan, dtype=np.float32)
    return arrays, offsets


def compute_subject_rdms(
    subject: str,
    target_ids: np.ndarray,
    behavior_path: Path,
    mask_paths: dict[str, Path],
    work_dir: Path,
    output_path: Path,
    *,
    roi_names: dict[int, str] | None = None,
    keep_temporary: bool = False,
) -> dict[str, object]:
    """Download one participant's sessions and write three condensed RDMs."""
    roi_names = dict(DEFAULT_ROIS if roi_names is None else roi_names)
    roi_ids = tuple(sorted(roi_names))
    behavior = pd.read_csv(behavior_path, sep="\t", usecols=["SESSION", "73KID"])
    schedule = build_schedule(behavior, target_ids)
    masks = {hemi: load_mask(mask_paths[hemi]) for hemi in ("lh", "rh")}
    if masks["lh"].shape != masks["rh"].shape:
        raise ValueError("left and right stream masks have different vertex counts")
    repeats, offsets = _allocate_repeats(target_ids.size, masks, roi_ids)
    union_indices = {
        hemi: np.flatnonzero(np.isin(mask, roi_ids)) for hemi, mask in masks.items()
    }
    local_roi_indices = {
        (hemi, roi_id): np.flatnonzero(masks[hemi][union_indices[hemi]] == roi_id)
        for hemi in ("lh", "rh")
        for roi_id in roi_ids
    }

    subject_work = work_dir / subject
    subject_work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    downloaded_bytes = 0
    for session, target_trials in sorted(schedule.items()):
        destinations = {
            hemi: subject_work / f"{hemi}.betas_session{session:02d}.mgh"
            for hemi in ("lh", "rh")
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    _download_with_retries,
                    BETAS_URL.format(subject=subject, hemi=hemi, session=session),
                    destination,
                ): hemi
                for hemi, destination in destinations.items()
            }
            for future in as_completed(futures):
                future.result()
        trials_in_session = int(np.sum(behavior["SESSION"] == session))
        for hemi in ("lh", "rh"):
            beta_path = destinations[hemi]
            downloaded_bytes += beta_path.stat().st_size
            standardized = zscore_selected_vertices(beta_path, union_indices[hemi])
            if standardized.shape[1] != trials_in_session:
                raise ValueError(
                    f"{subject} session {session} {hemi}: beta trials "
                    f"{standardized.shape[1]} != behavior trials {trials_in_session}"
                )
            for roi_id in roi_ids:
                local = local_roi_indices[(hemi, roi_id)]
                destination = offsets[(hemi, roi_id)]
                for item in target_trials:
                    repeats[roi_id][item.repeat, item.target, destination] = standardized[
                        local, item.trial
                    ]
            del standardized
            if not keep_temporary:
                beta_path.unlink(missing_ok=True)

    payload: dict[str, np.ndarray] = {"nsd_ids": np.asarray(target_ids, dtype=np.int64)}
    roi_metadata: dict[str, dict[str, int]] = {}
    for roi_id in roi_ids:
        with np.errstate(invalid="ignore"):
            patterns = np.nanmean(repeats[roi_id], axis=0)
        finite_vertices = np.all(np.isfinite(patterns), axis=0)
        patterns = patterns[:, finite_vertices]
        if patterns.shape[1] < 2:
            raise ValueError(f"{subject} {roi_names[roi_id]} has fewer than two valid vertices")
        rdm = pdist(patterns, metric="correlation")
        if not np.all(np.isfinite(rdm)):
            raise ValueError(f"{subject} {roi_names[roi_id]} RDM contains non-finite values")
        payload[f"rdm__{roi_names[roi_id]}"] = rdm.astype(np.float32)
        roi_metadata[roi_names[roi_id]] = {
            "vertices_total": int(repeats[roi_id].shape[-1]),
            "vertices_retained": int(finite_vertices.sum()),
        }

    metadata: dict[str, object] = {
        "schema_version": 1,
        "subject": subject,
        "n_images": int(target_ids.size),
        "n_sessions_with_targets": len(schedule),
        "downloaded_bytes": downloaded_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "targetspace": "fsaverage",
        "session_standardization": "scipy.stats.zscore(axis=-1, ddof=0, nan_policy=propagate)",
        "repeat_aggregation": "numpy.nanmean across exactly three presentations",
        "rdm_metric": "correlation",
        "rois": roi_metadata,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    metadata["output"] = str(output_path)
    metadata["sha256"] = sha256(output_path)
    if not keep_temporary:
        try:
            subject_work.rmdir()
        except OSError:
            pass
    return metadata


def _subject_worker(arguments: tuple[object, ...]) -> dict[str, object]:
    return compute_subject_rdms(*arguments)


def run_all_subjects(
    index_path: Path,
    output_dir: Path,
    metadata_dir: Path,
    work_dir: Path,
    subjects: list[str],
    *,
    workers: int = 1,
    keep_temporary: bool = False,
) -> dict[str, object]:
    target_ids = read_nsd_ids(index_path)
    metadata_paths = prepare_metadata(metadata_dir, subjects)
    ctab = parse_ctab(metadata_paths["ctab"])
    roi_names = {roi_id: ctab.get(roi_id, name) for roi_id, name in DEFAULT_ROIS.items()}
    mask_paths = {hemi: metadata_paths[f"mask_{hemi}"] for hemi in ("lh", "rh")}
    jobs = [
        (
            subject,
            target_ids,
            metadata_paths[f"behavior_{subject}"],
            mask_paths,
            work_dir,
            output_dir / f"{subject}.npz",
        )
        for subject in subjects
    ]
    results: list[dict[str, object]] = []
    if workers == 1:
        for arguments in jobs:
            results.append(
                compute_subject_rdms(
                    *arguments, roi_names=roi_names, keep_temporary=keep_temporary
                )
            )
    else:
        expanded_jobs = [(*arguments, roi_names, keep_temporary) for arguments in jobs]
        # Positional wrapper keeps ProcessPool serialization simple and explicit.
        def submit(pool: ProcessPoolExecutor, arguments: tuple[object, ...]):
            base = arguments[:6]
            names = arguments[6]
            keep = arguments[7]
            return pool.submit(
                compute_subject_rdms,
                *base,
                roi_names=names,
                keep_temporary=keep,
            )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {submit(pool, arguments): str(arguments[0]) for arguments in expanded_jobs}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: str(item["subject"]))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "reference_pipeline": "adriendoerig/visuo_llm",
        "reference_commit": VISUO_LLM_COMMIT,
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "subjects": results,
        "workers": workers,
        "roi_id_to_name": {str(key): value for key, value in roi_names.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "raw_nsd_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("data/processed/shared515_index.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/neural_raw"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/raw_nsd_metadata"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--subject", action="append", dest="subjects")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-temporary", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    subjects = args.subjects or [f"subj{number:02d}" for number in range(1, 9)]
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="jspace-nsd-") as temporary:
            manifest = run_all_subjects(
                args.index,
                args.output_dir,
                args.metadata_dir,
                Path(temporary),
                subjects,
                workers=args.workers,
                keep_temporary=args.keep_temporary,
            )
    else:
        manifest = run_all_subjects(
            args.index,
            args.output_dir,
            args.metadata_dir,
            args.work_dir,
            subjects,
            workers=args.workers,
            keep_temporary=args.keep_temporary,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
