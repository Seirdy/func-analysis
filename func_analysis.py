#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""Analyzer of the behavior of mathematical functions.

Given functions that are real, continuous, and differentiable
across specified intervals, this module graphs the function and
performs calculus to determine:

1. Special points
    - roots
    - critical Number, relative/absolute extrema, and saddle points
    - points of inflection
2. Special intervals
    - intervals of increase/decrease
    - Intervals of concavity

Optional data can be provided to improve precision and performance.
These data can be:

- Any of the above data
- The number of any type of special point
- The first, second, and/or third derivatives of the function

This code uses the Black formatter.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import mpmath as mp
import numpy as np
from pandas import DataFrame
from scipy.optimize import brentq

__version__ = "0.0.1"
# the three main types of Number used in this program are:
#   1. mpmath arbitrary-precision floating points (mp.mpf)
#   2. numpy.float64
#   3. Python's native floats.
Number = Union[mp.mpf, np.float64, float]
Interval = Tuple[mp.mpf, mp.mpf]  # intervals between mp.mpf Number
Func = Callable[
    [Union[Iterable[Number], Number]], Union[Iterable[mp.mpf], mp.mpf]
]


def find_one_zero(
    func: Func, x_range: Tuple[Number, Number], starting_point: Number
) -> mp.mpf:
    """Find the zero of a function in a given interval.

    mpmath's zero-finding algorithms require a starting "guess" point.
    `scipy.optimize.brentq` can find an imprecise zero in a given
    interval. Combining these, this method uses scipy.optimize's output
    as a starting point for mpmath's more precise root-finding algo.

    If a starting point is provided, the interval argument
    becomes unnecessary.

    Args:
        func: The function to find a zero for.
        x_range: The x-interval in which to find a zero.
        starting_point: A guess-point. Can be `None`, in which case
            use `scipy.optimize.brentq` to calculate one.

    Returns:
        A single very precise zero.

    """
    # If a starting point is not provided, find one.
    if not starting_point and x_range:
        starting_point = brentq(
            f=func, a=x_range[0], b=x_range[1], maxiter=50, disp=False
        )
    return mp.findroot(f=func, x0=starting_point)


def assemble_table(
    func: Callable[[Iterable[mp.mpf]], Iterable[mp.mpf]],
    x_vals: Iterable[Number],
) -> np.ndarray:
    """Make a table of values for the function with the given x-vals.

    Args:
        func: The function to generate an x-y table for.
        x_vals: The values to put in the x-column of the table.

    Returns:
        A 2d numpy array containing a column of x-values (see
        Args: x_vals) and computed y-values.

    """
    y_vals = func(x_vals)
    return np.stack((x_vals, y_vals), axis=-1)


class AnalyzedFunc:
    """Parent class of all function analysis."""

    def __init__(
        self,
        func: Func,
        x_range: Tuple[Number, Number],
        derivatives: Dict[int, Func] = None,
    ):
        """Initialize the object.

        Args:
            func: The function
            x_range: The interval of x-values. This is treated as an
                open interval except when finding absolute extrema.
            derivatives: A dictionary of derivatives. derivatives[nth]
                is the nth derivative of func.
        """
        self._func = mp.memoize(func)
        self.x_range = x_range
        self.min_x: Number = self.x_range[0]
        self.max_x: Number = self.x_range[1]

        self._derivatives: Dict[int, Callable[[mp.mpf], mp.mpf]]
        if derivatives:
            self._derivatives = {
                k: mp.memoize(v) for k, v in derivatives.items()
            }
            self._derivatives = derivatives
        else:
            self._derivatives = {}

        # A table of x- and y-values saved as an np.ndarray.
        self.func_iterable(self.x_range)

    def func(self, x_vals):
        """Define the function to be analyzed.

        self._func might already be able to handle an iterable input,
        in which case this method acts like a different name for the
        same thing. Otherwise, this method maps self._func over
        iterable input.

        Args:
            x_vals: One or more x-values.

        Returns:
            One or more y-values. If x_vals type is Iterable[Number],
            return type is Iterable[mp.mpf]. If x_vals type is Number,
            return type is mp.mpf.

        """
        try:
            return self._func(x_vals)
        except TypeError:
            return [self._func(x_val) for x_val in x_vals]

    def func_iterable(self, x_vals: Iterable[Number]) -> Iterable[mp.mpf]:
        """Map self.func over iterable input.

        This also saves x- and y- values in self.plotted_points.

        Args:
            x_vals: Input values for self.func. See doc for
                AnalyzedFunc.func()

        Returns:
            self.func(x_valus). See doc for AnalyzedFunc.func().

        """
        y_vals = self.func(x_vals)
        # build np.ndarray of new coords
        result_array: np.ndarray = np.stack((x_vals, y_vals), axis=-1)
        # attach this np.ndarray to self.plotted_points.
        # Create self.plotted_points if it doesn't already exist.
        try:
            self.plotted_points: np.ndarray = np.concatenate(
                (self.plotted_points, result_array)
            )
        except AttributeError:
            self.plotted_points = result_array
        return y_vals

    def plot(self, points_to_plot: int) -> np.ndarray:
        """Produce x,y pairs for self.func in range.

        Args:
            points_to_plot: The number of evenly-spaced points to plot
                in self.x_range.

        Returns:
            A 2d numpy array containing a column of x-values (see
            Args: x_vals) and computed y-values.

        """
        x_vals = np.linspace(self.min_x, self.max_x, points_to_plot)
        y_vals = self.func_iterable(x_vals)
        return np.stack((x_vals, y_vals), axis=-1)

    def nth_derivative(self, nth: int) -> Callable[[mp.mpf], mp.mpf]:
        """Create the nth-derivative of a function.

        If the nth-derivative has already been found, grab it.
        Otherwise, Numerically compute an arbitrary derivative of
        self.func and save it for re-use.

        Args:
            nth: The derivative desired.

        Returns:
            The nth-derivative of the function.

        """
        try:
            return self._derivatives[nth]
        except KeyError:
            if nth == 0:
                return self.func

            def derivative_computed(
                x_val: Number
            ) -> Callable[[Number], Number]:
                """Evaluate derivatives at an input value.

                Args:
                    x_val: Input value to the nth-derivative of
                        self.func.

                Returns:
                    The nth-derivative of self.func.

                """
                return mp.diff(self.func, x_val, n=nth)

            # Add this to the dictionary
            self._derivatives[nth] = derivative_computed
            return self._derivatives[nth]

    def has_symmetry(self, axis: mp.mpf) -> bool:
        """Determine if func is symmetric about given axis.

        Args:
            axis: The number representing the domain of the vertical
                line about which self.func has symmetry.

        Returns:
            bool: True if self.func is symmetric about axis, False
            otherwise.

        """
        try:
            # Dedupe self.plotted_points.
            # pylint: disable=attribute-defined-outside-init
            # Pandas is really good at deduplication,
            # so we'll use a DataFrame object.
            self.plotted_points = (
                DataFrame(self.plotted_points).drop_duplicates().values
            )  # pylint: enable=attribute-defined-outside-init
            if len(self.plotted_points) < 50:
                self.plot(50)
        except AttributeError:
            self.plot(50)
        x_mirror = np.subtract(2 * axis, self.plotted_points[:, 0])
        return np.array_equal(
            self.plotted_points[:, 1], self.func_iterable(x_mirror)
        )


def _zero_intervals(coordinate_pairs: np.ndarray) -> List[Interval]:
    """Find open intervals containing zeros.

    Args:
        coordinate_pairs: An x-y table represented by a 2d ndarray.

    Returns:
        A list of x-intervals across which self.func crosses the
        x-axis

    """
    y_vals = coordinate_pairs[:, 1]
    x_vals = coordinate_pairs[:, 0]
    # First determine if each coordinate is above the x-axis.
    is_positive = y_vals > 0
    # Using is_positive, return a list of tuples containing every pair of
    # consecutive x-values that has correspondin y-values on the opposite
    # sides of the x-axis
    return [
        (x_vals[i], x_vals[i + 1])
        for i in range(0, len(coordinate_pairs) - 1)
        if is_positive[i] is not is_positive[i + 1]
    ]


class FuncZeros(AnalyzedFunc):
    """A function with some of its properties.

    This object calculates and saves roots of the function.
    """

    def __init__(
        self, zeros_wanted: int, known_zeros: Iterable[Number] = None, **kwargs
    ):
        """Initialize the object.

        Args:
            zeros_wanted: The number
            known_zeros: List of zeros already known. Used as starting
                points for more exact computation.
            **kwargs: Keyward arguments to pass to super. See doc for
                AnalyzedFunc.__init__()
        """
        super().__init__(**kwargs)
        self.zeros_wanted = zeros_wanted
        if known_zeros:
            self._known_zeros = known_zeros

    def _zeros_in_range(self) -> np.ndarray:
        """Filter self._known_zeros to contain just zeros in range.

        Returns:
            A subset of self._known_zeros that includes only values in
            self.x_range

        """
        try:
            known_zeros = np.array(self._known_zeros)
            bools = np.logical_and(
                self.min_x <= known_zeros, self.max_x >= known_zeros
            )
            return known_zeros[bools]
        except AttributeError:
            return np.array([])

    def _all_zero_intervals(self) -> List[Interval]:
        """Find ALL zero intervals for this object's function.

        Returns:
            All x-intervals across which self.func crosses the x-axis.
            Minimum number of intervals is self.zeros_wanted.

        """
        points_to_plot = self.zeros_wanted + 3

        zero_intervals_found: List[Interval] = _zero_intervals(
            self.plot(points_to_plot)
        )
        while len(zero_intervals_found) < self.zeros_wanted:
            points_to_plot += 1
            zero_intervals_found = _zero_intervals(self.plot(points_to_plot))
        return zero_intervals_found

    def _solved_intervals(self) -> List[Interval]:
        """Filter zero intervals containing a zero already known.

        Returns:
            A subset of self._all_zero_intervals() containing zeros
            in self._known_zeros

        """
        # There are none if there are no zeros already known.
        intervals_found: List[Tuple[mp.mpf, mp.mpf]] = []
        zeros_found = self._zeros_in_range()
        if not zeros_found.size:
            return intervals_found
        # We're only looking at what's in the window specified.
        available_zero_intervals = self._all_zero_intervals()
        for possible_zero_interval in available_zero_intervals:
            # if any zeros are found that fit in this interval,
            # append this interval.
            if np.logical_and(
                zeros_found > possible_zero_interval[0],
                zeros_found < possible_zero_interval[1],
            ).any():
                intervals_found.append(possible_zero_interval)
        return intervals_found

    def zeros(self) -> np.ndarray:
        """List all zeros wanted in x_range.

        Returns:
            An array of precise zeros for self.func.

        """
        # starting_points is a list of any zeros already found.
        # These zeros are imprecise starting points for exact
        # computation.
        starting_points = self._zeros_in_range()
        # Intervals containing these zeros.
        intervals_with_zero = self._solved_intervals()
        sp_index: int = 0
        # The list of zeros we'll put together. It starts empty.
        zeros: List[mp.mpf] = []
        for x_interval in self._all_zero_intervals():
            # mpmath's root-finders can take an imprecise starting point.
            # If this interval has an already-found zero,
            # use that as the starting point. Otherwise, let
            # find_one_zero() use the interval's bounds to find a zero.
            starting_pt: Number = None
            if x_interval in intervals_with_zero:
                starting_pt = starting_points[sp_index]
                sp_index += 1
            # Add the exact zero.
            zeros.append(find_one_zero(self.func, x_interval, starting_pt))
        return np.array(zeros)


class FuncSpecialPts(FuncZeros):
    """A RootedFunction with additional properties (critical Number).

    This object includes a function and its properties. If those
    properties are not provided, they will be calculated and saved.

    The function properties included:
        - generator of derivatives
        - zeros
        - critical Number
    """

    def __init__(
        self,
        crits_wanted: int = None,
        known_crits: Tuple[Number, ...] = None,
        pois_wanted: int = None,
        known_pois: Tuple[Number, ...] = None,
        **kwargs,
    ):
        """Initialize a CriticalFunction.

        Args:
            crits_wanted: Number of critical nums to calculate.
            known_crits: A list of critical numbers already known,
                used as starting points for more precise calculation.
            pois_wanted: Number of points of inflection to calculate.
            known_pois: A list of points of inflection already known,
                used as starting points for more precise calculation.
            **kwargs: Keyword arguments to pass to super. See
                doc for FuncZeros.__init__()
        """
        super().__init__(**kwargs)
        if not crits_wanted:
            self.crits_wanted = self.zeros_wanted - 1
        else:
            self.crits_wanted = crits_wanted
        if pois_wanted is None:
            self.pois_wanted = self.crits_wanted - 1
        else:
            self.pois_wanted = pois_wanted
        self.known_crits = known_crits
        self.known_pois = known_pois

    # pylint: disable=undefined-variable
    def rooted_first_derivative(
        self
    ) -> Union[FuncZeros, FuncSpecialPts]:  # NOQA: F821
        """Return FuncZeros object for self.func's 1st derivative.

        Returns:
            FuncZeros: Analyzed 1st derivative of self.func, complete
            with zeros, crits, and an iterable func.

        """
        # pylint: enable=undefined-variable
        derivatives_of_fprime: Optional[Dict[int, Callable[[mp.mpf], mp.mpf]]]
        derivatives_of_fprime = {
            nth - 1: self._derivatives[nth] for nth in self._derivatives.keys()
        }
        return FuncSpecialPts(
            func=self.nth_derivative(1),
            zeros_wanted=max(self.crits_wanted, 1),
            known_zeros=self.known_crits,
            derivatives=derivatives_of_fprime,
            x_range=self.x_range,
            crits_wanted=self.pois_wanted,
            known_crits=self.known_pois,
        )

    def rooted_second_derivative(self) -> FuncZeros:
        """Return FuncZeros object for self.func's 2nd derivative.

        Returns:
            FuncZeros: Analyzed 2nd derivative of self.func, complete
            with zeros and an iterable func.

        """
        derivatives_of_fprime2: Optional[Dict[int, Callable[[mp.mpf], mp.mpf]]]
        derivatives_of_fprime2 = {
            nth - 2: self._derivatives[nth] for nth in self._derivatives.keys()
        }
        return FuncZeros(
            func=self.nth_derivative(2),
            zeros_wanted=max(self.pois_wanted, 1),
            known_zeros=self.known_pois,
            derivatives=derivatives_of_fprime2,
            x_range=self.x_range,
        )

    def crits(self) -> np.ndarray:
        """List all critical numbers wanted.

        This works by returning the zeros of the 1st derivative.

        Returns:
            An array of precise critical points for self.func.

        """
        return self.rooted_first_derivative().zeros()

    def pois(self) -> np.ndarray:
        """List all points of inflection wanted.

        Returns:
            An array of precise points of inflection for self.func.

        """
        fp2_zeros = self.rooted_second_derivative().zeros()
        return fp2_zeros[
            np.array(self.rooted_first_derivative().func_iterable(fp2_zeros))
            != 0
        ]

    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        Returns:
            A list of x-values for vertical lines about which self.func
            has symmetry.

        """
        return [crit for crit in self.crits() if self.has_symmetry(axis=crit)]


def _make_intervals(points: List[Number]) -> List[Interval]:
    """Pair each point to the next.

    Args:
        points: A list of points

    Returns:
        A list of intervals in which every two points have been paired.

    """
    return [(points[i], points[i + 1]) for i in range(0, len(points) - 1)]


def _increasing_intervals(
    func: Callable[[mp.mpf], mp.mpf], intervals: List[Interval]
) -> List[Interval]:
    """Return intervals across which func is decreasing.

    Args:
        func: The function to analyze.
        intervals: List of x-intervals to filter.

    Returns:
        Subset of intervals containing only intervals across which
        self.func is increasing.

    """
    return [
        x_interval
        for x_interval in intervals
        if func(x_interval[0]) < func(x_interval[1])
    ]


def _decreasing_intervals(
    func: Callable[[mp.mpf], mp.mpf], intervals: List[Interval]
) -> List[Interval]:
    """Return intervals across which func is decreasing.

    Args:
        func: The function to analyze.
        intervals: List of x-intervals to filter.

    Returns:
        Subset of intervals containing only intervals across which
        self.func is decreasing.

    """
    return [
        x_interval
        for x_interval in intervals
        if func(x_interval[0]) > func(x_interval[1])
    ]


class FuncIntervals(FuncSpecialPts):
    """Complete function analysis, with special points and intervals.

    Intervals found:
        - Intervals of increase/decrease
        - Concavity/convexity
    """

    def _construct_intervals(self, points) -> List[Interval]:
        points.insert(0, self.min_x)
        points.append(self.max_x)
        return _make_intervals(points)

    def increasing(self) -> List[Interval]:
        """List self.func's intervals of increase.

        Returns:
            All intervals of self.x_range across which self.func is
            increasing.

        """
        return _increasing_intervals(
            self.func, self._construct_intervals(list(self.crits()))
        )

    def decreasing(self) -> List[Interval]:
        """List self.func's intervals of decrease.

        Returns:
            All intervals of self.x_range across which self.func is
            increasing.

        """
        return _decreasing_intervals(
            self.func, self._construct_intervals(list(self.crits()))
        )

    def concave(self) -> List[Interval]:
        """List self.func's intervals of concavity (opening up).

        Returns:
            All intervals of self.x_range across which self.func is
            concave (opening up).

        """
        return _increasing_intervals(
            self.rooted_first_derivative().func,
            self._construct_intervals(list(self.pois())),
        )

    def convex(self) -> List[Interval]:
        """List self.func's intervals of convexity. (opening down).

        Returns:
            All intervals of self.x_range across which self.func is
            convex (opening down).

        """
        return _decreasing_intervals(
            self.rooted_first_derivative().func,
            self._construct_intervals(list(self.pois())),
        )

    def relative_maxima(self) -> np.ndarray:
        """List all relative maxima of self.func.

        Find the subset of self.crits() that includes critical numbers
        appearing on intervals in which func is convex.

        Returns:
            Array of precise relative maxima of self.func appearing in
            x_range.

        """
        crits_found = self.crits()
        mask = np.array(self.rooted_second_derivative().func(crits_found)) < 0
        return crits_found[mask]

    def relative_minima(self) -> np.ndarray:
        """List all relative maxima of self.func.

        Find the subset of self.crits() that includes critical numbers
        appearing on intervals in which func is concave.

        Returns:
            Array of precise relative minima of self.func appearing in
            x_range.

        """
        crits_found = self.crits()
        mask = np.array(self.rooted_second_derivative().func(crits_found)) > 0
        return crits_found[mask]

    def absolute_maximum(self) -> Iterable[mp.mpf]:
        """Find the absolute maximum of self.simple_func.

        Find the maximum of self.relative_maxima and the bounds of
        x_range.

        Returns:
            The x-y coordinate of the absolute maximum of self.func in
            the form [x, y].

        """
        x_vals: List[mp.mpf] = np.concatenate(
            (self.relative_maxima(), self.x_range)
        )
        pairs: np.ndarray = assemble_table(self.func_iterable, x_vals)
        return pairs[np.argmax(pairs[:, 1])]

    def absolute_minimum(self) -> Iterable[mp.mpf]:
        """Find the absolute minimum of self.simple_func.

        Find the minimum of self.relative_minima and the bounds of
        x_range.

        Returns:
            The x-y coordinate of the absolute minimum of self.func in
            the form [x, y].

        """
        x_vals: List[mp.mpf] = np.concatenate(
            (self.relative_minima(), self.x_range)
        )
        pairs: np.ndarray = assemble_table(self.func_iterable, x_vals)
        return pairs[np.argmin(pairs[:, 1])]
