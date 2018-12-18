"""The classes that do the actual function analysis."""

from __future__ import annotations

from collections import abc
from numbers import Real
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import mpmath as mp
import numpy as np

from ._decorators import SaveXY, singledispatchmethod
from ._util import (
    assemble_table,
    decreasing_intervals,
    find_one_zero,
    increasing_intervals,
    items_in_range,
    make_intervals,
    zero_intervals,
)

Interval = Tuple[mp.mpf, mp.mpf]  # intervals between mp.mpf numbers
Func = Callable[[Union[Iterable[Real], Real]], Union[Iterable[mp.mpf], mp.mpf]]


class AnalyzedFuncBase:
    """Parent class of all function analysis."""

    def __init__(
        self,
        func: Func,
        x_range: Tuple[Real, Real],
        derivatives: Dict[int, Func] = None,
    ):
        """Initialize the object.

        Parameters
        ----------
        func
            The function
        x_range
            The interval of x-values. This is treated as an
            open interval except when finding absolute extrema.
        derivatives
            A dictionary of derivatives. derivatives[nth]
            is the nth derivative of func.

        """
        self._func_plotted = SaveXY(func)
        self._func = mp.memoize(self._func_plotted)
        self.x_range = x_range
        self.min_x: Real = min(self.x_range)
        self.max_x: Real = max(self.x_range)

        self._derivatives: Dict[int, Callable[[mp.mpf], mp.mpf]]
        if derivatives:
            self._derivatives = {
                k: mp.memoize(v) for k, v in derivatives.items()
            }
        else:
            self._derivatives = {}

        # A table of x- and y-values saved as an np.ndarray.
        self.func(self.x_range)

    @singledispatchmethod
    def func(self, x_val: Real) -> mp.mpf:
        """Define the function to be analyzed.

        Parameters
        ----------
        x_val
            The independent variable to input to self._func.

        Returns
        -------
        y_val
            The y_value of self._func when x is x_val

        """
        return self._func(x_val)

    @func.register(abc.Iterable)
    def _(self, x_vals: Iterable[Real]) -> Iterable[mp.mpf]:
        """Register an iterable type as the parameter for self.func.

        Map self._func over iterable input.

        Parameters
        ----------
        x_vals
            One or more x-values.

        Returns
        -------
        y_vals
            One or more y-values. If x_vals type is Iterable[Real],
            return type is Iterable[mp.mpf]. If x_vals type is Real,
            return type is mp.mpf.

        """
        return [self.func(x_val) for x_val in x_vals]

    del _

    def plot(self, points_to_plot: int) -> np.ndarray:
        """Produce x,y pairs for self.func in range.

        Parameters
        ----------
        points_to_plot
            The number of evenly-spaced points to plot in self.x_range.

        Returns
        -------
        xy_table : np.ndarray
            A 2d numpy array containing a column of x-values (see
            Args: x_vals) and computed y-values.

        """
        x_vals = np.linspace(*self.x_range, points_to_plot)
        y_vals = self.func(x_vals)
        return np.stack((x_vals, y_vals), axis=-1)

    def nth_derivative(self, nth: int) -> Callable[[mp.mpf], mp.mpf]:
        """Create the nth-derivative of a function.

        If the nth-derivative has already been found, grab it.
        Otherwise, numerically compute an arbitrary derivative of
        self.func and save it for re-use.

        Parameters
        ----------
        nth
            The derivative desired.

        Returns
        -------
        Callable[[mp.mpf], mp.mpf]
            The nth-derivative of the function.

        """
        try:
            return self._derivatives[nth]
        except KeyError:
            if nth == 0:
                return self.func
            return lambda x_val: mp.diff(self.func, x_val, n=nth)

    def has_symmetry(self, axis: mp.mpf) -> bool:
        """Determine if func is symmetric about given axis.

        Parameters
        ----------
        axis
            The number representing the domain of the vertical
            line about which self.func has symmetry.

        Returns
        -------
        bool
            True if self.func is symmetric about axis, False otherwise.

        """
        try:
            assert len(self._func_plotted.plotted_points) > 50
        except (AssertionError, AttributeError):
            self.plot(50)
        saved_coordinates = np.array(self._func_plotted.plotted_points)
        x_vals = saved_coordinates[:, 0]
        y_vals = saved_coordinates[:, 1]
        x_mirror = np.subtract(2 * axis, x_vals)
        y_mirror = self.func(x_mirror)
        return np.array_equal(np.abs(y_vals), np.abs(y_mirror))


class FuncZeros(AnalyzedFuncBase):
    """A function with some of its properties.

    This object calculates and saves roots of the function.
    """

    def __init__(
        self,
        zeros_wanted: int = 0,
        known_zeros: Iterable[Real] = None,
        **kwargs,
    ):
        """Initialize the object.

        Parameters
        ----------
        zeros_wanted
            The number of zeros to find
        known_zeros
            List of zeros already known. Used as starting points for
            more exact computation.
        **kwargs
            Keyword arguments to pass to super. See doc for
            AnalyzedFunc.__init__()

        """
        super().__init__(**kwargs)
        self.zeros_wanted = zeros_wanted
        if known_zeros is not None:
            self._zeros: np.ndarray = items_in_range(
                np.array(known_zeros), self.x_range
            )
        else:
            self._zeros = None

    def _all_zero_intervals(self) -> List[Interval]:
        """Find ALL zero intervals for this object's function.

        Returns
        -------
        List[Interval]
            All x-intervals across which self.func crosses the x-axis.
            Minimum number of intervals is self.zeros_wanted.

        """
        points_to_plot = self.zeros_wanted + 3

        zero_intervals_found: List[Interval] = zero_intervals(
            self.plot(points_to_plot)
        )
        while len(zero_intervals_found) < self.zeros_wanted:
            points_to_plot += 1
            zero_intervals_found = zero_intervals(self.plot(points_to_plot))
        return zero_intervals_found

    def _solved_intervals(self) -> List[Interval]:
        """Filter zero intervals containing a zero already known.

        Returns
        -------
        filtered_intervals : List[Interval]
            A subset of self._all_zero_intervals() containing zeros
            in self._zeros

        """
        # There are none if there are no zeros already known.
        intervals_found: List[Tuple[mp.mpf, mp.mpf]] = []
        zeros_found = self._zeros
        if zeros_found is None or not zeros_found.size:
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

    def _compute_zeros(self) -> np.ndarray:
        """Compute all zeros wanted and updates self._zeros."""
        # starting_points is a list of any zeros already found.
        # These zeros are imprecise starting points for exact
        # computation.
        starting_points = self._zeros
        # Intervals containing these zeros.
        intervals_with_zero = self._solved_intervals()
        sp_index = 0
        # The list of zeros we'll put together. It starts empty.
        zeros: List[mp.mpf] = []
        for x_interval in self._all_zero_intervals():
            # mpmath's root-finders can take an imprecise starting point.
            # If this interval has an already-found zero,
            # use that as the starting point. Otherwise, let
            # find_one_zero() use the interval's bounds to find a zero.
            starting_pt: Optional[Real] = None
            if x_interval in intervals_with_zero:
                starting_pt = starting_points[sp_index]
                sp_index += 1
            # Add the exact zero.
            zeros.append(find_one_zero(self.func, x_interval, starting_pt))
        return np.array(zeros)

    @property
    def zeros(self) -> np.ndarray:
        """List all zeros wanted in x_range.

        Returns
        -------
        np.ndarray
            An array of precise zeros for self.func.

        """
        if self._zeros is None or len(self._zeros) < self.zeros_wanted:
            self._zeros = self._compute_zeros()
        return self._zeros


class FuncSpecialPts(FuncZeros):
    """A RootedFunction with additional properties (critical Real).

    This object includes a function and its properties. If those
    properties are not provided, they will be calculated and saved.

    The function properties included:
        - generator of derivatives
        - zeros
        - critical Real
    """

    def __init__(
        self,
        crits_wanted: int = None,
        known_crits: Tuple[Real, ...] = None,
        pois_wanted: int = None,
        known_pois: Tuple[Real, ...] = None,
        **kwargs,
    ):
        """Initialize a CriticalFunction.

        Parameters
        ----------
        crits_wanted
            Real of critical nums to calculate.
        known_crits
            A list of critical numbers already known, used
            as starting points for more precise calculation.
        pois_wanted
            Real of points of inflection to calculate.
        known_pois
            A list of points of inflection already known, used
            as starting points for more precise calculation.
        **kwargs
            Keyword arguments to pass to super. See doc for
            FuncZeros.__init__()

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
        self._crits = known_crits
        self._pois = known_pois

    # pylint: disable=undefined-variable
    def rooted_first_derivative(self) -> FuncSpecialPts:  # noqa: F821
        """Return FuncZeros object for self.func's 1st derivative.

        Returns
        -------
        fprime : FuncZeros
            Analyzed 1st derivative of self.func, complete with zeros,
            crits, and an iterable func.

        """
        # pylint: enable=undefined-variable
        derivatives_of_fprime: Optional[
            Dict[int, Callable[[mp.mpf], mp.mpf]]
        ] = {
            nth - 1: self._derivatives[nth] for nth in self._derivatives.keys()
        }
        return FuncSpecialPts(
            func=self.nth_derivative(1),
            zeros_wanted=max(self.crits_wanted, 1),
            known_zeros=self._crits,
            derivatives=derivatives_of_fprime,
            x_range=self.x_range,
            crits_wanted=self.pois_wanted,
            known_crits=self._pois,
        )

    def rooted_second_derivative(self) -> FuncZeros:
        """Return FuncZeros object for self.func's 2nd derivative.

        Returns
        -------
        fprime2 : FuncZeros
            Analyzed 2nd derivative of self.func, complete with zeros
            and an iterable func.

        """
        derivatives_of_fprime2: Optional[
            Dict[int, Callable[[mp.mpf], mp.mpf]]
        ] = {
            nth - 2: self._derivatives[nth] for nth in self._derivatives.keys()
        }
        return FuncZeros(
            func=self.nth_derivative(2),
            zeros_wanted=max(self.pois_wanted, 1),
            known_zeros=self._pois,
            derivatives=derivatives_of_fprime2,
            x_range=self.x_range,
        )

    @property
    def crits(self) -> np.ndarray:
        """List all critical numbers wanted.

        This works by returning the zeros of the 1st derivative.

        Returns
        -------
        np.ndarray
            An array of precise critical points for self.func.

        """
        if self._crits is None or len(self._crits) < self.crits_wanted:
            self._crits = self.rooted_first_derivative().zeros
        return self._crits

    @property
    def pois(self) -> np.ndarray:
        """List all points of inflection wanted.

        Returns
        -------
        np.ndarray
            An array of precise points of inflection for self.func.

        """
        if self._pois is None or len(self._pois) < self.pois_wanted:
            fp2_zeros = self.rooted_second_derivative().zeros
            self._pois = fp2_zeros[
                np.array(self.rooted_first_derivative().func(fp2_zeros)) != 0
            ]
        return self._pois

    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        Returns
        -------
        list_of_axes : List[mpf]
            A list of x-values for vertical lines about which self.func
            has symmetry.

        """
        return [crit for crit in self.crits if self.has_symmetry(axis=crit)]


class AnalyzedFunc(FuncSpecialPts):
    """Complete function analysis, with special points and intervals.

    Intervals found:
        - Intervals of increase/decrease
        - Concavity/convexity
    """

    def _construct_intervals(self, points: List[Real]) -> List[Interval]:
        points.insert(0, self.min_x)
        points.append(self.max_x)
        return make_intervals(points)

    def increasing(self) -> List[Interval]:
        """List self.func's intervals of increase.

        Returns
        -------
        intervals_of_increase : List[Interval]
            All intervals of self.x_range across which self.func is
            increasing.

        """
        return increasing_intervals(
            self.func, self._construct_intervals(list(self.crits))
        )

    def decreasing(self) -> List[Interval]:
        """List self.func's intervals of decrease.

        Returns
        -------
        intervals_of_decrease : List[Interval]
            All intervals of self.x_range across which self.func is
            increasing.

        """
        return decreasing_intervals(
            self.func, self._construct_intervals(list(self.crits))
        )

    def concave(self) -> List[Interval]:
        """List self.func's intervals of concavity (opening up).

        Returns
        -------
        intervals_of_concavity : [Interval]
            All intervals of self.x_range across which self.func is
            concave (opening up).

        """
        return increasing_intervals(
            self.rooted_first_derivative().func,
            self._construct_intervals(list(self.pois)),
        )

    def convex(self) -> List[Interval]:
        """List self.func's intervals of convexity. (opening down).

        Returns
        -------
        intervals_of_convexity : List[Interval]
            All intervals of self.x_range across which self.func is
            convex (opening down).

        """
        return decreasing_intervals(
            self.rooted_first_derivative().func,
            self._construct_intervals(list(self.pois)),
        )

    def relative_maxima(self) -> np.ndarray:
        """List all relative maxima of self.func.

        Find the subset of self.crits that includes critical numbers
        appearing on intervals in which func is convex.

        Returns
        -------
        np.ndarray
            Array of precise relative maxima of self.func appearing in
            x_range.

        """
        crits_found = self.crits
        mask = np.array(self.rooted_second_derivative().func(crits_found)) < 0
        return crits_found[mask]

    def relative_minima(self) -> np.ndarray:
        """List all relative maxima of self.func.

        Find the subset of self.crits that includes critical numbers
        appearing on intervals in which func is concave.

        Returns
        -------
        np.ndarray
            Array of precise relative minima of self.func appearing in
            x_range.

        """
        crits_found = self.crits
        mask = np.array(self.rooted_second_derivative().func(crits_found)) > 0
        return crits_found[mask]

    def absolute_maximum(self) -> np.ndarray:
        """Find the absolute maximum of self.simple_func.

        Find the maximum of self.relative_maxima and the bounds of
        x_range.

        Returns
        -------
        abs_max : np.ndarray
            The x-y coordinate of the absolute maximum of self.func in
            the form [x, y].

        """
        x_vals: List[mp.mpf] = np.concatenate(
            (self.relative_maxima(), self.x_range)
        )
        pairs: np.ndarray = assemble_table(self.func, x_vals)
        return pairs[np.argmax(pairs[:, 1])]

    def absolute_minimum(self) -> np.ndarray:
        """Find the absolute minimum of self.simple_func.

        Find the minimum of self.relative_minima and the bounds of
        x_range.

        Returns
        -------
        abs_min : ndarray
            The x-y coordinate of the absolute minimum of self.func in
            the form [x, y].

        """
        x_vals: List[mp.mpf] = np.concatenate(
            (self.relative_minima(), self.x_range)
        )
        pairs: np.ndarray = assemble_table(self.func, x_vals)
        return pairs[np.argmin(pairs[:, 1])]
