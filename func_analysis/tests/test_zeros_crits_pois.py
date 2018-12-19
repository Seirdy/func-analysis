#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests zero-, crit-, and poi-finding algos in func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

import numpy as np

from tests import constants

from .._analysis_classes import AnalyzedFunc
from .helpers import assert_output_lessthan, mpf_assert_allclose, typecheck_zcp


def test_trig_func_has_correct_zeros(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.zeros."""
    typecheck_zcp(analyzed_trig_func.zeros)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.zeros),
        constants.TRIG_FUNC_ZEROS,
        rtol=constants.EPSILON_1,
    )
    # Does the function evaluate to 0 at its zeros?
    assert_output_lessthan(
        func=analyzed_trig_func.func,
        x_vals=analyzed_trig_func.zeros,
        max_y=3.5692e-19,
    )


def test_parabola_has_correct_zeros(analyzed_parab):
    """Check that analyzed_parab.zeros returns correct value."""
    np.testing.assert_equal(analyzed_parab.zeros, np.array([-2, 2]))


def test_incdecfunc_has_correct_zeros(analyzed_incdecfunc):
    """Test analyzed_incdecfunc.zeros returns correct value."""
    assert analyzed_incdecfunc.zeros == [-1]


def test_trig_func_has_correct_crits(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.crits."""
    typecheck_zcp(analyzed_trig_func.crits)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.crits),
        constants.TRIG_FUNC_CRITS,
        rtol=constants.EPSILON_1,
    )
    assert_output_lessthan(
        func=analyzed_trig_func.rooted_first_derivative().func,
        x_vals=analyzed_trig_func.crits,
        max_y=constants.EPSILON_1,
    )


def assert_trig_func_pois_are_accurate(analyzedfunc, pois_found: np.ndarray):
    """Test pois() accuracy."""
    assert (
        np.float128(pois_found[3] + 46.944_940_655_832_212_248_274_091_985_22)
        < constants.EPSILON_1
    )
    assert_output_lessthan(
        func=analyzedfunc.rooted_second_derivative().func,
        x_vals=analyzedfunc.pois,
        max_y=constants.EPSILON_1,
    )


def pois_stay_close_when_given_fp2(analyzedfunc, fp2_zeros):
    """Test pois() when providing second derivative.

    This makes sure that it is possible to provide a second derivative
    to AnalyzedFunc instances and that it gets used to improve
    accuracy.
    """
    analyzed_trig_func_with_fp2 = AnalyzedFunc(
        func=analyzedfunc.func,
        x_range=(-47.05, -46.3499),
        zeros_wanted=21,
        crits_wanted=21,
        known_zeros=[-47.038_289_673_236_127, -46.406_755_885_040_056],
        derivatives={2: fp2_zeros.func},
    )
    # make sure fp2_zeros.func() is actually used by tracking its call count
    fp2_zeros_func_counts_before = len(fp2_zeros.plotted_points)

    more_exact_pois = analyzed_trig_func_with_fp2.pois
    fp2_zeros_func_counts_after = len(fp2_zeros.plotted_points)
    assert fp2_zeros_func_counts_after - fp2_zeros_func_counts_before > 50

    typecheck_zcp(more_exact_pois)
    mpf_assert_allclose(fp2_zeros.zeros, more_exact_pois, constants.EPSILON_0)
    mpf_assert_allclose(
        more_exact_pois, analyzedfunc.pois, constants.EPSILON_2
    )


def test_trig_func_has_correct_pois(analyzed_trig_func, fp2_zeros):
    """Test the correctness of analyzed_trig_func.pois.

    First, compare the output with approximate floating-point values.
    Then, compare the output with the pois found from its exact second
    derivative.
    """
    # typechecking
    typecheck_zcp(analyzed_trig_func.pois)
    assert_trig_func_pois_are_accurate(
        analyzed_trig_func, analyzed_trig_func.pois
    )
    mpf_assert_allclose(
        fp2_zeros.zeros, analyzed_trig_func.pois, constants.EPSILON_2
    )
    pois_stay_close_when_given_fp2(analyzed_trig_func, fp2_zeros)


def test_parabola_has_correct_crits(analyzed_parab):
    """Check that analyzed_parab.crits returns correct value."""
    assert analyzed_parab.crits == [0]
