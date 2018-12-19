#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests inflection-point-finding algorithms in func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

import numpy as np

from func_analysis import AnalyzedFunc
from func_analysis.tests import constants, testing_utils


def assert_trig_func_pois_are_accurate(analyzedfunc, pois_found: np.ndarray):
    """Test pois() accuracy."""
    assert (
        np.float128(pois_found[3] + 46.94494065583221224827409198522)
        < constants.EPSILON1
    )
    testing_utils.assert_output_lessthan(
        func=analyzedfunc.rooted_second_derivative().func,
        x_vals=analyzedfunc.pois,
        max_y=constants.EPSILON1,
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
        known_zeros=[-47.038289673236127, -46.406755885040056],
        derivatives={2: fp2_zeros.func},
    )
    # make sure fp2_zeros.func() is actually used by tracking its call count
    fp2_zeros_func_counts_before = len(fp2_zeros.plotted_points)

    more_exact_pois = analyzed_trig_func_with_fp2.pois
    fp2_zeros_func_counts_after = len(fp2_zeros.plotted_points)
    assert fp2_zeros_func_counts_after - fp2_zeros_func_counts_before > 50

    testing_utils.typecheck_zcp(more_exact_pois)
    testing_utils.mpf_assert_allclose(
        fp2_zeros.zeros, more_exact_pois, constants.EPSILON0
    )
    testing_utils.mpf_assert_allclose(
        more_exact_pois, analyzedfunc.pois, constants.EPSILON2
    )


def test_trig_func_has_correct_pois(analyzed_trig_func, fp2_zeros):
    """Test the correctness of analyzed_trig_func.pois.

    First, compare the output with approximate floating-point values.
    Then, compare the output with the pois found from its exact second
    derivative.
    """
    # typechecking
    testing_utils.typecheck_zcp(analyzed_trig_func.pois)
    assert_trig_func_pois_are_accurate(
        analyzed_trig_func, analyzed_trig_func.pois
    )
    testing_utils.mpf_assert_allclose(
        fp2_zeros.zeros, analyzed_trig_func.pois, constants.EPSILON2
    )
    pois_stay_close_when_given_fp2(analyzed_trig_func, fp2_zeros)
