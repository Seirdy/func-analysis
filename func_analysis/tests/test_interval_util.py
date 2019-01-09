# -*- coding: utf-8 -*-
"""Tests for func_analysis._util."""

from numbers import Real
from typing import List

from numpy import float64

import pytest
from func_analysis.custom_types import Interval
from func_analysis.interval_util import (
    decreasing_intervals,
    increasing_intervals,
    make_intervals,
)


@pytest.fixture(scope="module")
def intervals() -> List:
    """Points for interval functions in _util."""
    sample_numbers = (-2, 8, -3, -4, -9, 12, 18, 4, 0)
    float_numbers: List[Real] = list(float64(sample_numbers))
    return list(make_intervals(float_numbers))


def test_make_intervals(intervals):
    """Test many helper functions that FuncIntervals leverages.

    These functions include make_pairs(), _increasing_intervals(),
    and _decreasing_intervals().
    """
    expected_values = (
        (-2, 8),
        (8, -3),
        (-3, -4),
        (-4, -9),
        (-9, 12),
        (12, 18),
        (18, 4),
        (4, 0),
    )
    # mypy thinks *Tuple[int, int] is incompatible with expected Real, Real
    expected = [Interval(*pair) for pair in expected_values]  # type: ignore
    actual = intervals

    assert expected == actual


def test_increasing_intervals(intervals):
    """Test increasing_intervals on a sample set of intervals."""
    expected = [intervals[0], intervals[4], intervals[5]]
    actual = increasing_intervals(lambda anything: anything, intervals)

    assert expected == actual


def test_decreasing_intervals(intervals):
    """Test decreasing_intervals on a sample set of intervals."""
    expected = [
        intervals[1],
        intervals[2],
        intervals[3],
        intervals[6],
        intervals[7],
    ]
    actual = decreasing_intervals(lambda anything: anything, intervals)
    assert expected == actual
