"""Test memoization of analyzed functions."""
from typing import Tuple

import numpy as np

from func_analysis.tests.call_counting import (
    counts_pre_analysis,
    workout_analyzed_func,
)


def test_original_funcs_never_called():
    """Ensure that funcs to be analyzed never get called directly."""
    counts = counts_pre_analysis()
    assert not np.count_nonzero(counts)


def test_call_counting(analyzed_trig_func):
    """Check and print call all_counts for each executed function."""
    counts = workout_analyzed_func(analyzed_trig_func)
    original_vals: Tuple[int, ...] = tuple(counts[0].values())
    deduped_vals: Tuple[int, ...] = tuple(counts[1].values())
    uniqueness = np.divide(deduped_vals, original_vals)
    # Ensure that memoized func isn't called again for repeat vals.
    counts_after_repeat = counts[0]["dupe"]
    assert original_vals.count(counts_after_repeat) > 1
    assert counts_after_repeat < 1700
    assert np.amin(uniqueness) > 0.8
