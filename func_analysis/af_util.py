"""Utilities for use in AnalyzedFunc."""


from numbers import Real
from typing import Callable, Iterable, List, NamedTuple, Sequence

import mpmath as mp
import numpy as np

Func = Callable[[Real], Real]


class Interval(NamedTuple):
    """Special NamedTuple for interval between two numbers."""

    start: Real
    stop: Real


class Coordinate(NamedTuple):
    """Special NamedTuple for x-y coordinate."""

    x_val: Real
    y_val: Real


def assemble_table(
    func: Callable[[Iterable[Real]], Iterable[Real]], x_vals: Iterable[Real]
) -> np.ndarray:
    """Make a table of values for the function with the given x-vals.

    Parameters
    ----------
    func
        The function to generate an x-y table for.
    x_vals
        The values to put in the x-column of the table.

    Returns
    -------
    np.ndarray
        A 2d numpy array containing a column of x-values (see
        Args: x_vals) and computed y-values.

    """
    y_vals = func(x_vals)
    return np.stack((x_vals, y_vals), axis=-1)


def zero_intervals(coordinate_pairs: np.ndarray) -> List[Interval]:
    """Find open intervals containing zeros.

    Parameters
    ----------
    coordinate_pairs
        An x-y table represented by a 2d ndarray.

    Returns
    -------
    List[Interval]
        A list of x-intervals across which self.func crosses the
        x-axis

    """
    y_vals = coordinate_pairs[:, 1]
    x_vals = coordinate_pairs[:, 0]
    # First determine if each coordinate is above the x-axis.
    is_positive = np.greater(y_vals, 0)
    # Using is_positive, return a list of tuples containing every pair of
    # consecutive x-values that has corresponding y-values on the opposite
    # sides of the x-axis
    return [
        Interval(x_vals[index], x_vals[index + 1])
        for index in range(0, len(coordinate_pairs) - 1)
        if is_positive[index] is not is_positive[index + 1]
    ]


def make_intervals(points: Sequence[Real]) -> List[Interval]:
    """Pair each point to the next.

    Parameters
    ----------
    points
        A list of points

    Returns
    -------
    List[Interval]
        A list of intervals in which every two points have been paired.

    """
    return [
        Interval(points[index], points[index + 1])
        for index in range(0, len(points) - 1)
    ]


def increasing_intervals(
    func: Callable[[Real], Real], intervals: List[Interval]
) -> List[Interval]:
    """Return intervals across which func is decreasing.

    Parameters
    ----------
    func
        The function to analyze.
    intervals
        List of x-intervals to filter.

    Returns
    -------
    List[Interval]
        Subset of intervals containing only intervals across which
        self.func is increasing.

    """
    return [
        x_interval
        for x_interval in intervals
        if func(x_interval[0]) < func(x_interval[1])
    ]


def decreasing_intervals(
    func: Callable[[mp.mpf], mp.mpf], intervals: List[Interval]
) -> List[Interval]:
    """Return intervals across which func is decreasing.

    Parameters
    ----------
    func
        The function to analyze.
    intervals
        List of x-intervals to filter.

    Returns
    -------
    List[Interval]
        Subset of intervals containing only intervals across which
        self.func is decreasing.

    """
    return [
        x_interval
        for x_interval in intervals
        if func(x_interval[0]) > func(x_interval[1])
    ]