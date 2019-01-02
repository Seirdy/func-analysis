"""Utilities for use in AnalyzedFunc."""
import itertools as it
from numbers import Real
from typing import Any, Callable, Iterable, Iterator, List, NamedTuple, Tuple

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


def zero_intervals(coordinates: np.ndarray) -> List[Interval]:
    """Find open intervals containing zeros.

    Parameters
    ----------
    coordinates
        An x-y table represented by a 2d ndarray.

    Returns
    -------
    List[Interval]
        A list of x-intervals across which self.func crosses the
        x-axis

    """
    x_intervals = make_intervals(coordinates[:, 0])
    is_positive = _make_pairs(np.greater(coordinates[:, 1], 0))
    return [
        interval_map[0]
        for interval_map in zip(x_intervals, is_positive)
        if interval_map[1][0] is not interval_map[1][1]
    ]


def _make_pairs(points: Iterable[Any]) -> Iterator[Tuple[Any, Any]]:
    """Pair each point to the next.

    Parameters
    ----------
    points
        A list of points

    Yields
    ------
    Pair: Tuple[Any, Any]
        Pairing of every two points as an Interval, with redundancy.

    """
    # Make an iterator that yields each point twice.
    doubled = it.chain.from_iterable((point, point) for point in points)
    # Chop off the first point. The last point will be dropped
    # automatically and zip two copies to form intervals.
    for pair in zip(*[it.islice(doubled, 1, None)] * 2):
        yield pair


def make_intervals(points: Iterable[Real]) -> Iterator[Interval]:
    """Make intervals that pair each point to the next.

    Parameters
    ----------
    points
        A list of points

    Yields
    ------
    Interval
        Pairing of every two points as an Interval, with redundancy.

    """
    for pair in _make_pairs(points):
        yield Interval(*pair)


def increasing_intervals(
    func: Callable[[Real], Real], intervals: Iterable[Interval]
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
        if func(x_interval.start) < func(x_interval.stop)
    ]


def decreasing_intervals(
    func: Callable[[mp.mpf], mp.mpf], intervals: Iterable[Interval]
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
        if func(x_interval.start) > func(x_interval.stop)
    ]
