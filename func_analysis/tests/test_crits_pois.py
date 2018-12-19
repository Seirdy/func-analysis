#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests crit- and poi-finding algos in func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

import numpy as np

from .._analysis_classes import AnalyzedFunc
from .helpers import assert_output_lessthan, mpf_assert_allclose, typecheck_zcp

EPSILON_0 = 1e-20
EPSILON_1 = 3.05e-15
EPSILON_2 = 1.196_789_1e-6


def test_trig_func_has_correct_crits(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.crits."""
    typecheck_zcp(analyzed_trig_func.crits)
    # approximate accuracy
    expected_crits = [
        -47.028_400_867_252_13,
        -46.995_216_177_440_79,
        -46.961_552_135_996_854,
        -46.928_318_300_227_15,
        -46.894_608_617_023_465,
        -46.861_324_416_365_34,
        -46.827_569_901_478_4,
        -46.794_234_116_960_42,
        -46.760_435_575_283_8,
        -46.727_046_992_482_36,
        -46.693_205_219_082_984,
        -46.659_762_632_756_91,
        -46.625_878_408_195_66,
        -46.592_380_626_945_825,
        -46.558_454_712_583_135,
        -46.524_900_563_516_226,
        -46.490_933_696_823_73,
        -46.457_322_030_198_725,
        -46.423_314_920_098_63,
        -46.389_644_613_934_3,
        -46.355_597_936_188_15,
    ]
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.crits), expected_crits, rtol=EPSILON_1
    )
    assert_output_lessthan(
        func=analyzed_trig_func.rooted_first_derivative().func,
        x_vals=analyzed_trig_func.crits,
        max_y=EPSILON_1,
    )


def assert_trig_func_pois_are_accurate(analyzedfunc, pois_found: np.ndarray):
    """Test pois() accuracy."""
    assert (
        np.float128(pois_found[3] + 46.944_940_655_832_212_248_274_091_985_22)
        < EPSILON_1
    )
    assert_output_lessthan(
        func=analyzedfunc.rooted_second_derivative().func,
        x_vals=analyzedfunc.pois,
        max_y=EPSILON_1,
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
    mpf_assert_allclose(fp2_zeros.zeros, more_exact_pois, EPSILON_0)
    mpf_assert_allclose(more_exact_pois, analyzedfunc.pois, EPSILON_2)


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
    mpf_assert_allclose(fp2_zeros.zeros, analyzed_trig_func.pois, EPSILON_2)
    pois_stay_close_when_given_fp2(analyzed_trig_func, fp2_zeros)


def test_parabola_has_correct_crits(analyzed_parab):
    """Check that analyzed_parab.crits returns correct value."""
    assert analyzed_parab.crits == [0]
