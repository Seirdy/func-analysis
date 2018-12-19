#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests root-finding algos in func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

import numpy as np

from .helpers import assert_output_lessthan, typecheck_zcp

EPSILON_0 = 1e-20
EPSILON_1 = 3.05e-15
EPSILON_2 = 1.196_789_1e-6


def test_trig_func_has_correct_zeros(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.zeros."""
    typecheck_zcp(analyzed_trig_func.zeros)
    expected_zeros = [
        -47.038_289_673_236_13,
        -47.018_473_233_395_28,
        -46.972_318_087_653_95,
        -46.950_739_626_397_91,
        -46.906_204_518_117_63,
        -46.882_958_270_910_02,
        -46.839_955_720_658_34,
        -46.815_121_707_485,
        -46.773_576_011_368_88,
        -46.747_224_922_729_01,
        -46.707_068_062_964_04,
        -46.679_264_553_080_85,
        -46.640_433_373_296_69,
        -46.611_238_416_225_63,
        -46.573_672_554_670_36,
        -46.543_145_221_101_68,
        -46.506_785_519_620_84,
        -46.474_984_380_574_83,
        -46.439_771_604_599_5,
        -46.406_755_885_040_05,
        -46.372_629_655_875_1,
    ]
    # approximate accuracy
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.zeros), expected_zeros, rtol=EPSILON_1
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
