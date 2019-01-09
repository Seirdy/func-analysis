# -*- coding: utf-8 -*-
"""Test that AnalyzedFunc handles no special pts being wanted."""
import numpy as np


def no_points_wanted(analyzed_func, found_attr: str):
    """Ensure no special points are returned when none are wanted."""
    wanted_attr = found_attr + "_wanted"
    setattr(analyzed_func, wanted_attr, 0)
    assert np.array_equal(getattr(analyzed_func, found_attr), np.array([]))


def test_no_points_wanted(all_analyzed_funcs):
    """Ensure no AnalyzedFunc returns unwanted special points."""
    for analyzed_func in all_analyzed_funcs:
        no_points_wanted(analyzed_func, "zeros")
        no_points_wanted(analyzed_func, "crits")
        no_points_wanted(analyzed_func, "pois")
