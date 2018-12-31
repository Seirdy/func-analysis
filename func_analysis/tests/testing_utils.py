"""Typechecking functions for testing AnalyzedFunc."""

import numpy as np


def mpf_assert_allclose(actual, desired, atol=1e-3):
    """Assert that the two arrays are close enough.

    Similar to numpy.testing.assert_allclose(), but specifically
    written for mpmath numbers.
    """
    assert np.amax(np.abs(np.subtract(actual, desired))) < atol


def assert_output_lessthan(func, x_vals, max_y):
    """Assert that func(x) < max_y for all x_vals."""
    y_vals = func(x_vals)
    assert np.amax(np.abs(y_vals)) < max_y


def calculate_error(expected, actual):
    """Calculate the experimental error."""
    return abs(1 - actual / expected)


def assert_error_lessthan(expected, actual, maxerror):
    """Assert that the error is within bounds."""
    error = calculate_error(expected, actual)
    assert error < maxerror
