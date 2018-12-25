"""Fixtures to represent sample AnalyzedFunc instances."""

import pytest
from func_analysis import AnalyzedFunc
from func_analysis.tests.call_counting import AnalyzedFuncCounted
from func_analysis.tests.funcs_to_analyze import (
    inc_dec_func,
    parab_func,
    sec_der,
    trig_func,
)

trig_func_args = {
    "func": trig_func,
    "x_range": (-47.05, -46.3499),
    "zeros_wanted": 21,
    "crits_wanted": 21,
    "known_zeros": [-47.038289673236127, -46.406755885040056],
}


@pytest.fixture
def analyzed_trig_func():
    """Fixture for an AnalyzedFunc describing trig_func."""
    analyzed_func = AnalyzedFunc(**trig_func_args)
    return analyzed_func


@pytest.fixture
def analyzed_trig_func_counted():
    """Version of analyzed_trig_func with counted calls."""
    analyzed_func = AnalyzedFuncCounted(**trig_func_args)
    return analyzed_func


@pytest.fixture
def fp2_zeros():
    """Fixture for an AnalyzedFunc describing sec_der."""
    return AnalyzedFunc(
        func=sec_der, x_range=(-47.05, -46.35), zeros_wanted=21
    )


@pytest.fixture
def analyzed_parab():
    """Fixture for an AnalyzedFunc describing parab_func."""
    return AnalyzedFunc(func=parab_func, x_range=(-8, 8), zeros_wanted=2)


@pytest.fixture
def analyzed_incdecfunc():
    """Fixture for an AnalyzedFunc describing inc_dec_func."""
    return AnalyzedFunc(
        func=inc_dec_func, x_range=(-3, -0.001), crits_wanted=1, zeros_wanted=1
    )
