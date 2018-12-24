"""Extend AnalyzedFuncBase to be able to use special points.

These include zeros, critical points, and points of inflection.
"""

from __future__ import annotations

from numbers import Real
from typing import Dict, Iterable, List, Optional, Tuple

import mpmath as mp
import numpy as np

from func_analysis.analyzed_func.af_base import AnalyzedFuncBase
from func_analysis.util import (
    Func,
    Interval,
    find_one_zero,
    items_in_range,
    zero_intervals,
)


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
        intervals_found: List[Interval] = []
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
            # If this interval has an already-found zero
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
        if not self.zeros_wanted:
            return np.array([])
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
    @property
    def rooted_first_derivative(self) -> FuncSpecialPts:  # noqa: F821
        """Return FuncZeros object for self.func's 1st derivative.

        Returns
        -------
        fprime : FuncZeros
            Analyzed 1st derivative of self.func, complete with zeros,
            crits, and an iterable func.

        """
        # pylint: enable=undefined-variable
        derivatives_of_fprime: Optional[Dict[int, Func]] = {
            nth - 1: self.derivatives[nth] for nth in self.derivatives.keys()
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

    @property
    def rooted_second_derivative(self) -> FuncZeros:
        """Return FuncZeros object for self.func's 2nd derivative.

        Returns
        -------
        fprime2 : FuncZeros
            Analyzed 2nd derivative of self.func, complete with zeros
            and an iterable func.

        """
        derivatives_of_fprime2: Optional[Dict[int, Func]] = {
            nth - 2: self.derivatives[nth] for nth in self.derivatives.keys()
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
        if not self.crits_wanted:
            return np.array([])
        if self._crits is None or len(self._crits) < self.crits_wanted:
            self._crits = self.rooted_first_derivative.zeros
        return self._crits

    @property
    def pois(self) -> np.ndarray:
        """List all points of inflection wanted.

        Returns
        -------
        np.ndarray
            An array of precise points of inflection for self.func.

        """
        if not self.pois_wanted:
            return np.array([])
        if self._pois is None or len(self._pois) < self.pois_wanted:
            fp2_zeros = self.rooted_second_derivative.zeros
            self._pois = fp2_zeros[
                np.nonzero(self.rooted_first_derivative.func(fp2_zeros))
            ]
        return self._pois

    @property
    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        Returns
        -------
        list_of_axes : List[mpf]
            A list of x-values for vertical lines about which self.func
            has symmetry.

        """
        return [crit for crit in self.crits if self.has_symmetry(axis=crit)]
