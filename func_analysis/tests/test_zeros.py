# -*- coding: utf-8 -*-

"""Tests zero-finding algorithms in func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

import numpy as np

from func_analysis import AnalyzedFunc
from func_analysis.tests import constants, testing_utils, typechecking


def test_trig_func_has_correct_zeros(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.zeros."""
    typechecking.typecheck_zcp(analyzed_trig_func.zeros)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.zeros),
        constants.TRIG_FUNC_ZEROS,
        rtol=constants.EPSILON1,
    )
    # Does the function evaluate to 0 at its zeros?
    testing_utils.assert_output_lessthan(
        func=analyzed_trig_func.func,
        x_vals=analyzed_trig_func.zeros,
        max_y=2.14e-13,
    )


def test_trig_func_zeros_none_provided(analyzed_trig_func, trig_func_args):
    """Test that zeros stay the same when some are provided."""
    trig_func_args["zeros"] = None
    analyzed_trig_func_none_provided = AnalyzedFunc(**trig_func_args)

    testing_utils.mpf_assert_allclose(
        analyzed_trig_func.zeros,
        analyzed_trig_func_none_provided.zeros,
        atol=constants.EPSILON1,
    )


def test_parabola_has_correct_zeros(analyzed_parab):
    """Check that analyzed_parab.zeros returns correct value."""
    np.testing.assert_equal(analyzed_parab.zeros, np.array([-2, 2]))


def test_incdecfunc_has_correct_zeros(analyzed_incdecfunc):
    """Test analyzed_incdecfunc.zeros returns correct value."""
    assert analyzed_incdecfunc.zeros == [-1]
