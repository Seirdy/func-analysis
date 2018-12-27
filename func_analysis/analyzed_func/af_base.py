"""Create AnalyzedFuncBase.

AnalyzedFuncBase is a parent class of AnalyzedFunc that contains
all the analysis that you can find without calculating roots.
"""

from collections import abc
from numbers import Real
from typing import Dict, Iterable, List, Tuple

import mpmath as mp
import numpy as np

from func_analysis.decorators import SaveXY
from func_analysis.util import Coordinate, Func, Interval

try:
    from functools import singledispatchmethod  # type: ignore # noqa: Z435
except ImportError:
    from func_analysis.decorators import singledispatchmethod  # noqa: Z435


class _AnalyzedFuncBaseInit(object):
    """Initialize AnalyzedFuncBase basic properties."""

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
        self._x_range = x_range
        self._derivatives = derivatives

    @property
    def x_range(self) -> Interval:
        """Make self._x_range an Interval object."""
        return Interval(*self._x_range)

    @property
    def derivatives(self) -> Dict[int, Func]:
        """Return all known derivatives of self.func."""
        if self._derivatives:
            return {
                derivative: mp.memoize(func)
                for derivative, func in self._derivatives.items()
            }
        return {}


class _AnalyzedFuncBaseFunc(_AnalyzedFuncBaseInit):
    """Initialize single-dispatched AnalyzedFuncBase.func."""

    # pylint: disable=no-self-use
    @singledispatchmethod
    def func(self, *args) -> None:
        """Abstract dispatched function to be analyzed.

        Raises
        ------
        TypeError
            If called with argument that isn't an instance
            of Real or Iterable[Real]

        """
        raise TypeError("Unsupported type '{0}'".format(type(*args)))

    # pylint: enable=no-self-use

    @func.register
    def func_real(self, x_val: Real) -> Real:
        """Define the function to be analyzed.

        Parameters
        ----------
        x_val
            The independent variable to input to self._func.

        Returns
        -------
        y_val : Real
            The y_value of self._func when x is x_val

        """
        return self._func(x_val)

    @func.register(abc.Iterable)
    def func_iterable(self, x_vals: Iterable[Real]) -> List[Real]:
        """Register an iterable type as the parameter for self.func.

        Map self._func over iterable input.

        Parameters
        ----------
        x_vals
            Multiple x_vals to pass to self.func_real.

        Returns
        -------
        y_vals : List[Real]
            The y-values corresponding to x_vals.

        """
        return [self.func_real(x_val) for x_val in x_vals]


class AnalyzedFuncBase(_AnalyzedFuncBaseFunc):
    """Parent class of all function analysis.

    AnalyzedFuncBase performs all possible analysis that does NOT
    require any calculus.
    """

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
        y_vals = self.func_iterable(x_vals)
        return np.stack((x_vals, y_vals), axis=-1)

    def nth_derivative(self, nth: int) -> Func:
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
        Func
            The nth-derivative of the function.

        """
        try:
            return self.derivatives[nth]
        except KeyError:
            if not nth:
                return self.func
            return lambda x_val: mp.diff(self.func, x_val, n=nth)

    @property
    def plotted_points(self) -> List[Coordinate]:
        """List all the coordinates calculated.

        Returns
        -------
        List[Coordinate]
            A list of x-y coordinate pairs that have been found.

        """
        return self._func_plotted.plotted_points

    def plot_enough(self, points_to_plot=50):
        """Make plotted_points meet a minimum length."""
        num_coords_found = len(self.plotted_points)
        if num_coords_found < points_to_plot:
            self.plot(points_to_plot - num_coords_found)
        return self.plotted_points

    def has_symmetry(self, axis: Real) -> bool:
        """Determine if self.func is symmetric about given axis.

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
        saved_coordinates = np.array(self.plot_enough())
        x_vals = saved_coordinates[:, 0]
        y_vals = saved_coordinates[:, 1]
        x_mirror = np.subtract(2 * axis, x_vals)
        y_mirror = self.func_iterable(x_mirror)
        return np.array_equal(np.abs(y_vals), np.abs(y_mirror))
