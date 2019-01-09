# -*- coding: utf-8 -*-
"""Typechecking functions for testing AnalyzedFunc."""
from numbers import Real
from typing import Callable, Iterable, Sequence, Union

import numpy as np

Number = Union[float, Real]
NumArray = Union[Number, Sequence[Number], Sequence[Sequence[Number]]]


def mpf_assert_allclose(
    actual: NumArray, desired: NumArray, atol: Number = 1e-3
):
    """Assert that the two arrays are close enough.

    Similar to numpy.testing.assert_allclose(), but specifically
    written for mpmath numbers.

    Parameters
    ----------
    actual, desired
        The arrays to compare.
    atol
        Absolute tolerance of error between arrays.

    """
    assert np.amax(np.abs(np.subtract(actual, desired))) < atol


def assert_output_lessthan(
    func: Callable, x_vals: Iterable[Number], max_y: Number
):
    """Assert that func(x) < max_y for all x_vals.

    Parameters
    ----------
    func
        The function whose output will be compared with max_y.
    x_vals
        The input to func.
    max_y
        The upper-bound allowed for outputs of func.

    """
    y_vals = func(x_vals)
    assert np.amax(np.abs(y_vals)) < max_y


def calculate_error(expected: Number, actual: Number):
    """Calculate the experimental error.

    Parameters
    ----------
    expected, actual
        The values to compare. These should be as close as possible.

    """
    return abs(1 - actual / expected)


def assert_error_lessthan(expected: Number, actual: Number, maxerror: Number):
    """Assert that the error is within bounds.

    Parameters
    ----------
    expected, actual
        The values to compare. These should be as close as possible.
    maxerror
        The maximum error allowed between expected and actual.

    """
    error = calculate_error(expected, actual)
    assert error < maxerror
