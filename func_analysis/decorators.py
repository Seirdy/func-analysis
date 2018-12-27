"""Decorators to use in AnalyzedFunc."""

from functools import singledispatch, update_wrapper
from numbers import Real
from typing import Callable, List

from func_analysis.util import Coordinate, Func


# noinspection PyPep8Naming,PyUnresolvedReferences
class singledispatchmethod(object):  # NOSONAR
    """Single-dispatch generic method descriptor.

    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.

    Backported from
    https://github.com/python/cpython/blob/master/Lib/functools.py
    """

    def __init__(self, func: Callable):
        """Initialize with the func and its dispatcher."""

        self.dispatcher = singledispatch(func)
        self.func = func

    def register(self, cls, method=None):
        """generic_method.register(cls, func) -> func
        Registers a new implementation for the given *cls* on a
        *generic_method*.
        """
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls):
        def _method(*args, **kwargs):
            """Access single-dispatch method."""
            method = self.dispatcher.dispatch(args[0].__class__)
            return method.__get__(obj, cls)(*args, **kwargs)  # type: ignore

        _method.__isabstractmethod__ = (  # type: ignore
            self.__isabstractmethod__
        )
        _method.register = self.register  # type: ignore
        update_wrapper(_method, self.func)
        return _method

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, "__isabstractmethod__", False)


class SaveXY(object):
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

    def __init__(self, func: Func):
        """Update wrapper; this is a decorator.

        Parameters
        ----------
        func
            The function to save values for.

        """
        self.func = func
        update_wrapper(self, self.func)
        self.plotted_points: List[Coordinate] = []

    def __call__(self, x_val: Real):
        """Save the x-y coordinate before returning the y-value.

        Parameters
        ----------
        x_val
            The x-value of the coordinate and the input to self.func

        """
        y_val = self.func(x_val)
        coordinate = Coordinate(x_val, y_val)
        self.plotted_points.append(coordinate)
        return y_val
