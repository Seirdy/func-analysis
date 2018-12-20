#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests critical-number-finding algorithms in func_analysis.

This deliberately uses a function requiring a high degree of precision
"""


import numpy as np

from func_analysis.tests import constants, testing_utils


def test_trig_func_has_correct_crits(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.crits."""
    testing_utils.typecheck_zcp(analyzed_trig_func.crits)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.crits),
        constants.TRIG_FUNC_CRITS,
        rtol=constants.EPSILON1,
    )
    testing_utils.assert_output_lessthan(
        func=analyzed_trig_func.rooted_first_derivative.func,
        x_vals=analyzed_trig_func.crits,
        max_y=constants.EPSILON1,
    )


def test_parabola_has_correct_crits(analyzed_parab):
    """Check that analyzed_parab.crits returns correct value."""
    assert analyzed_parab.crits == [0]
