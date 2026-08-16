import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from jspace_fmri.analysis import analyse, load_and_merge_model_features
from jspace_fmri.import_visuo import convert
from jspace_fmri.nsd import shared_three_repeat_ids
from jspace_fmri.simulate import simulate


def test_synthetic_end_to_end_recovers_injected_high_level_advantage(tmp_path):
    source = tmp_path / "synthetic"
    simulate(source, n_images=48, n_subjects=8, seed=7)
    output = source / "analysis"
    manifest = analyse(
        source / "model_features.npz",
        source / "neural",
        output,
        primary_prompt="raw",
    )
    assert manifest["n_subjects"] == 8
    group = pd.read_csv(output / "group_contrasts.csv")
    row = group[
        (group["family"] == "confirmatory_component")
        & (group["roi"] == "high_level")
        & (group["location"] == "layer_02")
    ].iloc[0]
    assert row["mean_delta"] > 0
    assert row["p_signflip_two_sided"] <= 0.05
    with (output / "analysis_manifest.json").open() as handle:
        assert json.load(handle)["inference_unit"] == "participant"


def test_visuo_import_requires_explicit_trust_and_writes_safe_npz(tmp_path):
    index = tmp_path / "index.csv"
    pd.DataFrame({"nsd_id": [1, 2, 3]}).to_csv(index, index=False)
    payload = {"subj01": {"higher visual": np.asarray([0.1, 0.2, 0.3])}}
    source = tmp_path / "cache.pkl"
    with source.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="--trusted"):
        convert(source, index, tmp_path / "out", trusted=False)
    manifest = convert(source, index, tmp_path / "out", trusted=True)
    assert manifest["roi_name_map"]["higher_visual"] == "higher visual"
    with np.load(tmp_path / "out" / "subj01.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["nsd_ids"], [1, 2, 3])
        np.testing.assert_allclose(archive["rdm__higher_visual"], [0.1, 0.2, 0.3])


def test_shared_ids_are_intersection_of_exactly_three_repeats(tmp_path):
    paths = []
    rows = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    for subject, extra in enumerate(([4, 4, 4], [5, 5, 5], [6, 6]), start=1):
        path = tmp_path / f"subj{subject:02d}.tsv"
        pd.DataFrame({"73KID": rows + list(extra)}).to_csv(path, sep="\t", index=False)
        paths.append(path)
    assert shared_three_repeat_ids(paths) == [1, 2, 3]


def test_multiple_feature_archives_are_reordered_and_merged(tmp_path):
    first = tmp_path / "qwen.npz"
    second = tmp_path / "mpnet.npz"
    np.savez(
        first,
        nsd_ids=np.asarray([10, 20, 30]),
        raw__raw__layer_01=np.asarray([[1.0], [2.0], [3.0]]),
    )
    np.savez(
        second,
        nsd_ids=np.asarray([30, 10, 20]),
        mpnet__raw__sentence=np.asarray([[30.0], [10.0], [20.0]]),
    )
    ids, features = load_and_merge_model_features([first, second])
    np.testing.assert_array_equal(ids, [10, 20, 30])
    np.testing.assert_array_equal(
        features["mpnet__raw__sentence"].ravel(), [10.0, 20.0, 30.0]
    )


def test_precomputed_model_rdms_match_feature_analysis(tmp_path):
    source = tmp_path / "synthetic"
    simulate(source, n_images=24, n_subjects=8, seed=11)
    with np.load(source / "model_features.npz", allow_pickle=False) as archive:
        payload = {"nsd_ids": archive["nsd_ids"]}
        for key in archive.files:
            if key == "nsd_ids":
                continue
            payload[f"rdm__{key}"] = pdist(archive[key], metric="correlation")
    rdm_path = source / "model_rdms.npz"
    np.savez_compressed(rdm_path, **payload)

    feature_output = source / "feature_analysis"
    rdm_output = source / "rdm_analysis"
    analyse(source / "model_features.npz", source / "neural", feature_output)
    analyse(
        None,
        source / "neural",
        rdm_output,
        model_rdms_path=rdm_path,
    )
    feature_scores = pd.read_csv(feature_output / "rsa_scores.csv").sort_values(
        ["subject", "roi", "feature_key"]
    )
    rdm_scores = pd.read_csv(rdm_output / "rsa_scores.csv").sort_values(
        ["subject", "roi", "feature_key"]
    )
    np.testing.assert_allclose(feature_scores["rsa"], rdm_scores["rsa"], atol=1e-12)
