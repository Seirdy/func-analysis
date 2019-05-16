# -*- coding: utf-8 -*-

"""Custom types to use throughout function analysis.

These are all used in mypy type annotations, and Interval and
Coordinate are used as simple data structures as well.
"""
from numbers import Real
from typing import Callable, NamedTuple

Func = Callable[[Real], Real]


# "start" and "stop" are sometimes erroneously flagged as unresolved
# references in the class docstring.
class Interval(NamedTuple):
    """Special NamedTuple for interval between two numbers.

    Attributes
    ----------
    start : Real
        The lower/left-bound of the interval
    stop : Real
        The upper/right-bound of the interval

    """

    start: Real
    stop: Real


# "x_val" and "y_val" are sometimes erroneously flagged as unresolved
# references in the class docstring.
class Coordinate(NamedTuple):
    """Special NamedTuple for x-y coordinate.

    Attributes
    ----------
    x_val : Real
    y_val : Real

    """

    x_val: Real
    y_val: Real
