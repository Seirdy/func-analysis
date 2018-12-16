"""Utilities for use in AnalyzedFunc."""


from numbers import Real
from typing import Callable, Iterable, List, MutableSequence, Tuple, Union

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

Interval = Tuple[mp.mpf, mp.mpf]  # intervals between mp.mpf numbers
Func = Callable[[Union[Iterable[Real], Real]], Union[Iterable[mp.mpf], mp.mpf]]


def find_one_zero(
    func: Func, x_range: Tuple[Real, Real], starting_point: Real = None
) -> mp.mpf:
    """Find the zero of a function in a given interval.

    mpmath's zero-finding algorithms require a starting "guess" point.
    `scipy.optimize.brentq` can find an imprecise zero in a given
    interval. Combining these, this method uses scipy.optimize's output
    as a starting point for mpmath's more precise root-finding algo.

    If a starting point is provided, the interval argument
    becomes unnecessary.

    Parameters
    ----------
    func
        The function to find a zero for.
    x_range
        The x-interval in which to find a zero.
    starting_point
        A guess-point. Can be `None`, in which case
        use `scipy.optimize.brentq` to calculate one.

    Returns
    -------
    mp.mpf
        A single very precise zero.

    """
    # If a starting point is not provided, find one.
    if starting_point is None:
        starting_point = brentq(
            f=func, a=x_range[0], b=x_range[1], maxiter=50, disp=False
        )
    # Maybe this starting point is good enough.
    if func(starting_point) == 0:
        return starting_point
    return mp.findroot(f=func, x0=starting_point)


def assemble_table(
    func: Callable[[Iterable[mp.mpf]], Iterable[mp.mpf]],
    x_vals: Iterable[Real],
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
    is_positive = y_vals > 0
    # Using is_positive, return a list of tuples containing every pair of
    # consecutive x-values that has corresponding y-values on the opposite
    # sides of the x-axis
    return [
        (x_vals[i], x_vals[i + 1])
        for i in range(0, len(coordinate_pairs) - 1)
        if is_positive[i] is not is_positive[i + 1]
    ]


def items_in_range(
    items: np.ndarray, interval: Tuple[Real, Real]
) -> np.ndarray:
    """Filter items to contain just items in closed interval.

    Parameters
    ----------
    items
        The array to filter
    interval
        The closed interval of acceptable values.

    Returns
    -------
    filtered_items : np.ndarray
        A subset of items that includes only values in interval

    """
    mask = np.logical_and(min(interval) <= items, max(interval) >= items)
    return items[mask]


def make_intervals(points: MutableSequence[Real]) -> List[Interval]:
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
    return [(points[i], points[i + 1]) for i in range(0, len(points) - 1)]


def increasing_intervals(
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
