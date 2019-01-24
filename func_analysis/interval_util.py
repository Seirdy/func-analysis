# -*- coding: utf-8 -*-

"""Utilities for use in AnalyzedFunc."""
import itertools as it
from numbers import Real
from typing import Any, Iterable, Iterator, List, Tuple

from func_analysis.custom_types import Func, Interval


def make_pairs(points: Iterable[Any]) -> Iterator[Tuple[Any, Any]]:
    """Pair each point to the next.

    Parameters
    ----------
    points
        A list of points

    Yields
    ------
    Pair: Tuple[Any, Any]
        Pairing of every two points as a ``tuple``.

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
        Pairing of every two points as an ``Interval``.

    """
    for pair in make_pairs(points):
        yield Interval(*pair)


def increasing_intervals(
    func: Func, intervals: Iterable[Interval]
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
        ``func`` is increasing.

    """
    return [
        x_interval
        for x_interval in intervals
        if func(x_interval.start) < func(x_interval.stop)
    ]


def decreasing_intervals(
    func: Func, intervals: Iterable[Interval]
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
        ``func`` is decreasing.

    """
    return [
        x_interval
        for x_interval in intervals
        if func(x_interval.start) > func(x_interval.stop)
    ]
