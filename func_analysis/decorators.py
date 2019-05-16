# -*- coding: utf-8 -*-

"""Decorators to use in AnalyzedFunc."""

from functools import singledispatch, update_wrapper
from numbers import Real
from typing import Callable, List

from func_analysis.custom_types import Coordinate, Func


class singledispatchmethod(object):  # noqa: N801
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

    def register(self, cls, method=None):  # NOQA
        """Register method for decorated function.

        Registers a new implementation for the given *cls* on a
        *generic_method*.
        """
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls):  # NOQA
        """Retrieve decorated function."""

        def _method(*args, **kwargs):
            """Access single-dispatch method."""
            method = self.dispatcher.dispatch(args[0].__class__)
            return method.__get__(obj, cls)(  # noqa: Z
                *args, **kwargs  # type: ignore
            )

        _method.__isabstractmethod__ = (  # type: ignore  # noqa: Z
            self.__isabstractmethod__
        )
        _method.register = self.register  # type: ignore
        update_wrapper(_method, self.func)
        return _method

    @property
    def __isabstractmethod__(self):
        """Magic method for marking decorated func as implemented."""
        return getattr(self.func, "__isabstractmethod__", False)  # noqa: Z


class SaveXY(object):
    """Class decorator for saving X-Y coordinates.

    This is not used for memoization; mp.memoize() serves that purpose
    better because of how it handles mp.mpf numbers. This only exists
    to save values to use in AnalyzedFunc.has_symmetry.

    Attributes
    ----------
    func: Callable[[Real], mp.mpf]
        The function to decorate and save values for.
    plotted_points: List[Coordinate]
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


def copy_metadata_from(good_obj):
    """Copy another object's metadata into doc of decorated object.

    This copies the docstring and type annotations from one object to
    another.

    Parameters
    ----------
    good_obj
        The object to copy the metadata from.

    Returns
    -------
    updated_obj_wrapper : object
        ``bad_obj`` with metadata copied from ``good_obj``.

    """

    def wrapper(bad_obj):
        """Wrap the decorated object.

        Parameters
        ----------
        bad_obj
            The decorated object to update the docstring for.

        Returns
        -------
        updated_obj : object
            ``bad_doc_obj`` with a docstring copied from
            ``good_obj``.

        """
        update_wrapper(wrapper=bad_obj, wrapped=good_obj)
        return bad_obj

    return wrapper
