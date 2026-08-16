import json

import numpy as np

from jspace_fmri.analysis import load_model_rdms
from jspace_fmri.recovery import merge_recovery_archives
from jspace_fmri.rsa import expected_condensed_length


def test_merge_recovery_adds_rdm_prefix(tmp_path):
    ids = np.asarray([1, 2, 3, 4])
    n_pairs = expected_condensed_length(ids.size)
    source = tmp_path / "single.npz"
    output = tmp_path / "merged.npz"
    np.savez_compressed(
        source,
        nsd_ids=ids,
        manifest_json=np.asarray(json.dumps({"layer": 9})),
        raw__raw__layer_09=np.arange(n_pairs, dtype=float),
        jlens__raw__layer_09=np.arange(n_pairs, dtype=float) + 1,
    )
    merge_recovery_archives([source], output)
    loaded_ids, rdms = load_model_rdms(output)
    assert np.array_equal(loaded_ids, ids)
    assert set(rdms) == {"raw__raw__layer_09", "jlens__raw__layer_09"}
