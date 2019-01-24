# -*- coding: utf-8 -*-

"""The classes that do the actual function analysis."""

from __future__ import annotations

from numbers import Real
from typing import Callable, Iterator, List, MutableSequence, Sequence

import numpy as np

from func_analysis.analyzed_func.af_crits_pois import AnalyzedFuncSpecialPts
from func_analysis.custom_types import Coordinate, Interval
from func_analysis.interval_util import (
    decreasing_intervals,
    increasing_intervals,
    make_intervals,
)


class AnalyzedFuncIntervals(AnalyzedFuncSpecialPts):
    """Function analysis concerning special intervals.

    These intervals include:
        - Intervals of increase/decrease.
        - Intervals of concavity/convexity.
    """

    def _construct_intervals(
        self, points: MutableSequence[Real]
    ) -> Iterator[Interval]:
        """Construct intervals to filter in interval analysis.

        All interval analysis filters a set of intervals constructed
        from:

        - the upper- and lower- bounds of the x-range
        - a set of special points (critical or inflection points).

        This method constructs those intervals.

        Parameters
        ----------
        points
            A set of special points to serve as the bounds of intervals.

        Returns
        -------
        Iterator[Interval]
            A bunch of intervals to filter into
            increasing/decreasing/concave/convex intervals.

        """
        points.insert(0, self.x_range.start)
        points.append(self.x_range.stop)
        return make_intervals(points)

    @property
    def increasing(self) -> List[Interval]:
        """List the function's intervals of increase.

        Returns
        -------
        intervals_of_increase : List[Interval]
            All intervals within ``self.x_range`` across which
            the function is increasing.

        """
        return increasing_intervals(
            self.func, self._construct_intervals(list(self.crits))
        )

    @property
    def decreasing(self) -> List[Interval]:
        """List the function's intervals of decrease.

        Returns
        -------
        intervals_of_decrease : List[Interval]
            All intervals within ``self.x_range`` across which
            the function is increasing.

        """
        return decreasing_intervals(
            self.func, self._construct_intervals(list(self.crits))
        )

    @property
    def concave(self) -> List[Interval]:
        """List the function's intervals of concavity (opening up).

        Returns
        -------
        intervals_of_concavity : List[Interval]
            All intervals within ``self.x_range`` across which
            the function is concave (opening up).

        """
        return increasing_intervals(
            self.rooted_first_derivative.func,
            self._construct_intervals(list(self.pois)),
        )

    @property
    def convex(self) -> List[Interval]:
        """List the function's intervals of convexity. (opening down).

        Returns
        -------
        intervals_of_convexity : List[Interval]
            All intervals within ``self.x_range`` across which
            the function is convex (opening down).

        """
        return decreasing_intervals(
            self.rooted_first_derivative.func,
            self._construct_intervals(list(self.pois)),
        )


class AnalyzedFuncExtrema(object):
    """Function analysis concerning special points.

    This class adds relative/absolute extrema to the analysis.
    """

    def __init__(self, **kwargs):
        """Initialize the object."""
        self._af_specialpts = AnalyzedFuncSpecialPts(**kwargs)

    def _concavity_at_crits(self):
        """Find slope of second derivative at each critical point."""
        return self._af_specialpts.rooted_second_derivative.func_iterable(
            self._af_specialpts.crits
        )

    @property
    def relative_maxima(self) -> np.ndarray:
        """Find all relative minima of the function.

        Find the subset of ``self.crits`` containing critical numbers
        appearing on intervals in which the function is convex.

        Returns
        -------
        relative_maxima : ndarray of Reals
            Array of precise relative maxima appearing in x_range.

        """
        mask = np.less(self._concavity_at_crits(), 0)
        return self._af_specialpts.crits[mask]

    @property
    def relative_minima(self) -> np.ndarray:
        """Find all relative maxima of the function.

        Find the subset of ``crits`` containing critical numbers
        appearing on intervals in which the function is concave.

        Returns
        -------
        relative_minima : ndarray of Coordinates
            Array of precise relative minima appearing in x_range.

        """
        concavity = self._af_specialpts.rooted_second_derivative.func_iterable(
            self._af_specialpts.crits
        )
        mask = np.greater(concavity, 0)
        return self._af_specialpts.crits[mask]

    @property
    def absolute_maximum(self) -> Coordinate:
        """Find the absolute maximum of the function.

        Find the maximum of self.relative_maxima and the bounds of
        x_range.

        Returns
        -------
        abs_max : Coordinate
            The coordinate of the absolute maximum of the function.

        """
        return self._absolute_extrema(self.relative_maxima, np.argmax)

    @property
    def absolute_minimum(self) -> Coordinate:
        """Find the absolute minimum of self.simple_func.

        Find the minimum of self.relative_minima and the bounds of
        x_range.

        Returns
        -------
        abs_min : Coordinate
            The coordinate of the absolute minimum of the function.

        """
        return self._absolute_extrema(self.relative_minima, np.argmin)

    def _absolute_extrema(
        self,
        points: Sequence,
        extrema_finder: Callable[[np.ndarray], np.ndarray],
    ) -> Coordinate:
        """Generalized absolute-extrema finder.

        Parameters
        ----------
        points
            The contenders for the absolute extrema excluding the
            bounds of ``self.x_range``.
        extrema_finder
            The function that computes the index of the extrema
            (i.e., ``np.argmax`` or ``np.argmin``).

        Returns
        -------
        absolute_extrema : Coordinate
            The coordinates of the absolute extrema.

        """
        x_vals: np.ndarray = np.concatenate(
            (points, self._af_specialpts.x_range)
        )
        pairs: np.ndarray = np.stack(
            (x_vals, self._af_specialpts.func_iterable(x_vals)), axis=-1
        )
        # build a Coordinate object from the right item in pairs.
        return Coordinate(*pairs[extrema_finder(pairs[:, 1])])
