# -*- coding: utf-8 -*-

"""Create AnalyzedFuncBase.

AnalyzedFuncBase is a parent class of AnalyzedFunc that contains
all the analysis that you can find without calculating roots.
"""

from collections import abc
from numbers import Real
from typing import Dict, Iterable, List, Tuple

import mpmath as mp
import numpy as np

from func_analysis.custom_types import Coordinate, Func, Interval
from func_analysis.decorators import SaveXY

# Need to allow nested imports in order to use backport.
try:
    from functools import singledispatchmethod  # type: ignore # noqa: Z435
except ImportError:
    from func_analysis.decorators import singledispatchmethod  # noqa: Z435


class AnalyzedFuncBase(object):
    """Parent class of all function analysis.

    AnalyzedFuncBase performs all possible analysis that does NOT
    require any calculus.
    """

    def __init__(
        self,
        func: Func,
        x_range: Tuple[Real, Real],
        derivatives: Dict[int, Func] = None,
        **_,
    ):
        """Initialize AnalyzedFuncBase with explicit MRO.

        Parameters
        ----------
        func
            The function to analyze
        x_range
            The interval of x-values. This is treated as an
            open interval except when finding absolute extrema.
        derivatives
            A dictionary of derivatives. ``derivatives[nth]``
            is the nth derivative of the function.
        _
            Unused arguments intended for other ``AnalyzedFuncBase``
            children.

        """

        self.x_range = Interval(*x_range)
        self._derivatives = derivatives
        self.func_plotted = SaveXY(func)
        self.func_memoized = mp.memoize(self.func_plotted)

    # Before it gets dispatch methods, _AnalyzedFuncBaseFunc.func
    # doesn't access instance or class state; however, it still needs
    # to be in the class because registered implementations func_real
    # and func_iterable access instance state. Therefore, it makes
    # sense to violate wemake-python-styleguide's Z433 and make it
    # static.
    @singledispatchmethod  # noqa: Z433
    @staticmethod
    def func(*args) -> None:
        """Abstract dispatched function to be analyzed.

        Parameters
        ----------
        *args
            Bad arguments of unknown type.

        Raises
        ------
        TypeError
            If called with argument that isn't an instance
            of ``Real`` or ``Iterable[Real]``.

        """
        bad_types = (type(bad_arg) for bad_arg in args)
        raise TypeError(
            "Unsupported type '{0}'. ".format(*bad_types)
            + "Expected type abc.Real or Iterable[abc.Real]."
        )

    @func.register
    def func_real(self, x_val: Real) -> Real:
        """Define the function to be analyzed.

        Parameters
        ----------
        x_val
            The independent variable to pass to
            ``self.func_memoized``.

        Returns
        -------
        y_val : Real
            The y_value of ``self.func_memoized`` when given ``x_val``.

        """
        return self.func_memoized(x_val)

    @func.register(abc.Iterable)
    def func_iterable(self, x_vals: Iterable[Real]) -> List[Real]:
        """Register an iterable type for ``self.func``.

        Map ``self.func_real`` over iterable input.

        Parameters
        ----------
        x_vals
            Multiple x_vals to pass to ``self.func_real``.

        Returns
        -------
        y_vals : List[Real]
            The y-values corresponding to ``x_vals``.

        """
        return [self.func_real(x_val) for x_val in x_vals]

    @property
    def derivatives(self) -> Dict[int, Func]:
        """Return all known derivatives of the function.

        Returns
        -------
        derivatives : Dict[int, Callable[[Real], Real]
            A dictionary of known derivatives of the function being
            analyzed.

        """
        if self._derivatives:
            return {
                derivative: mp.memoize(func)
                for derivative, func in self._derivatives.items()
            }
        return {}

    def plot(self, points_to_plot: int) -> np.ndarray:
        """Produce x,y pairs for the function.

        Parameters
        ----------
        points_to_plot
            The number of evenly-spaced points to plot across
            ``self.x_range``.

        Returns
        -------
        xy_table : ndarray of ints
            A 2d numpy array containing one column for x-values and
            another column for computed y-values.

        """
        x_vals = np.linspace(*self.x_range, points_to_plot)
        y_vals = self.func_iterable(x_vals)
        return np.stack((x_vals, y_vals), axis=-1)

    def nth_derivative(self, nth: int) -> Func:
        """Create the nth-derivative of a function.

        If the nth-derivative has already been found, return that.
        Otherwise, numerically estimate an arbitrary derivative of
        the function.

        Parameters
        ----------
        nth
            The derivative desired.

        Returns
        -------
        nth_derivative : Func
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
        plotted_points : List[Coordinate]
            A list of x-y coordinate pairs that have been found.

        """
        return self.func_plotted.plotted_points
