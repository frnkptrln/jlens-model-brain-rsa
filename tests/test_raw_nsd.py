import numpy as np
import pandas as pd
import pytest

from jspace_fmri.raw_nsd import build_schedule


def test_build_schedule_uses_session_local_trials_and_repeat_slots():
    behavior = pd.DataFrame(
        {
            "SESSION": [1, 1, 1, 2, 2, 2, 3, 3],
            "73KID": [10, 99, 20, 20, 10, 99, 10, 20],
        }
    )
    schedule = build_schedule(behavior, np.asarray([10, 20]))
    assert [(x.trial, x.target, x.repeat) for x in schedule[1]] == [
        (0, 0, 0),
        (2, 1, 0),
    ]
    assert [(x.trial, x.target, x.repeat) for x in schedule[2]] == [
        (0, 1, 1),
        (1, 0, 1),
    ]
    assert [(x.trial, x.target, x.repeat) for x in schedule[3]] == [
        (0, 0, 2),
        (1, 1, 2),
    ]


def test_build_schedule_rejects_incomplete_repeats():
    behavior = pd.DataFrame({"SESSION": [1, 1, 2], "73KID": [10, 20, 10]})
    with pytest.raises(ValueError, match="exactly 3 repeats"):
        build_schedule(behavior, np.asarray([10, 20]))
