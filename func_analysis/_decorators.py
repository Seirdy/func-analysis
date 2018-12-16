"""Decorators to use in AnalyzedFunc."""

from functools import singledispatch, update_wrapper
from numbers import Real
from typing import Callable, List, Tuple

import mpmath as mp


def singledispatchmethod(func: Callable):
    """Single-dispatch generic method decorator."""
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        """Wrap decorated function."""
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register  # type: ignore
    update_wrapper(wrapper, func)
    return wrapper


class SaveXY:
    """Class decorator for saving X-Y coordinates.

    This is not used for memoization; mp.memoize() serves that purpose
    better because of how it handles mp.mpf numbers. This only exists
    to save values to use in AnalyzedFunc.has_symmetry.

    Attributes
    ----------
    func: Callable[[Real], mp.mpf]
        The function to decorate and save values for.
    plotted_points: List[Tuple[mp.mpf, mp.mpf]]
        The saved coordinate pairs.

    """

    def __init__(self, func: Callable[[Real], mp.mpf]):
        """Update wrapper; this is a decorator.

        Parameters
        ----------
        func
            The function to save values for.

        """
        self.func = func
        update_wrapper(self, self.func)
        self.plotted_points: List[Tuple[mp.mpf, mp.mpf]] = []

    def __call__(self, x_val: Real):
        """Save the x-y coordinate before returning the y-value.

        Parameters
        ----------
        x_val
            The x-value of the coordinate and the input to self.func

        """
        y_val = self.func(x_val)
        coordinate = (x_val, y_val)
        self.plotted_points.append(coordinate)
        return y_val
