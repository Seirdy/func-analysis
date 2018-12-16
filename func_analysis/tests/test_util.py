"""Tests for func_analysis._util."""

from typing import List

from .._util import decreasing_intervals, increasing_intervals, make_intervals


def test_interval_helpers_work_correctly():
    """Test many helper functions that FuncIntervals leverages.

    These functions include _make_intervals(), _increasing_intervals(),
    and _decreasing_intervals()/
    """
    points: List = [-2, 8, -3, -4, -9, 12, 18, 4, 0]
    expected_intervals: List = [
        (-2, 8),
        (8, -3),
        (-3, -4),
        (-4, -9),
        (-9, 12),
        (12, 18),
        (18, 4),
        (4, 0),
    ]
    assert make_intervals(points) == expected_intervals

    def dummy_func(x_val):
        """Return input.

        Used to test _increasing_intervals()
        and _decreasing_intervals().
        """
        return x_val

    assert increasing_intervals(dummy_func, make_intervals(points)) == [
        expected_intervals[0],
        expected_intervals[4],
        expected_intervals[5],
    ]
    assert decreasing_intervals(dummy_func, make_intervals(points)) == [
        expected_intervals[1],
        expected_intervals[2],
        expected_intervals[3],
        expected_intervals[6],
        expected_intervals[7],
    ]
