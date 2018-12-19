"""Tests for func_analysis._util."""

from numbers import Real
from typing import List

from numpy import float64

import pytest

from .._util import decreasing_intervals, increasing_intervals, make_intervals


@pytest.fixture()
def intervals() -> List:
    """Points for interval functions in _util."""
    points: List[Real] = list(float64([-2, 8, -3, -4, -9, 12, 18, 4, 0]))
    return make_intervals(points)


def test_make_intervals(intervals):
    """Test many helper functions that FuncIntervals leverages.

    These functions include _make_intervals(), _increasing_intervals(),
    and _decreasing_intervals()/
    """
    expected: List = [
        (-2, 8),
        (8, -3),
        (-3, -4),
        (-4, -9),
        (-9, 12),
        (12, 18),
        (18, 4),
        (4, 0),
    ]
    actual = intervals

    assert expected == actual


def test_increasing_intervals(intervals):
    """Test increasing_intervals on a sample set of intervals."""
    expected = [intervals[0], intervals[4], intervals[5]]
    actual = increasing_intervals(lambda x: x, intervals)

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
    actual = decreasing_intervals(lambda x: x, intervals)
    assert expected == actual
