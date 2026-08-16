# Data contract

No NSD images or fMRI data are committed to this repository.

`jspace-prepare-nsd` downloads only public experimental metadata, participant
response tables, and COCO caption annotations. It writes:

- `processed/shared515_captions.jsonl`: one record per shared image;
- `processed/shared515_index.csv`: the ordered NSD/COCO identifiers;
- `processed/shared515_manifest.json`: source URLs, hashes, and counts.

The JSONL schema is:

```json
{"nsd_id": 123, "nsd_index": 122, "coco_id": 456, "coco_split": "train2017", "captions": ["..."]}
```

`nsd_id` is the one-based `73KID` used by NSD response files. `nsd_index` is
the corresponding zero-based row in `nsd_stim_info_merged.csv`.

## Neural data accepted by the RSA runner

Use one `.npz` file per participant. Every file must contain `nsd_ids` in the
same one-based convention. An ROI can be supplied either as:

- `patterns__ROI_NAME`: `[n_images, n_voxels]` response patterns; or
- `rdm__ROI_NAME`: a condensed RDM of length `n_images * (n_images - 1) / 2`.

The recommended shortcut is to convert the trusted, local cache produced by
`visuo_llm`:

```text
results_dir/roi_analyses/streams_roi_results_correlation/cache/
  streams_all_neural_rdms_correlation_special515.pkl
```

Run `jspace-import-visuo --help` for the conversion command. Pickle input must
only be accepted from a trusted collaborator because loading pickle can execute
code.

