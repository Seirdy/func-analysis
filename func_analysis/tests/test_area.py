"""Test area-finding methods."""

from func_analysis.tests import constants


def signed_area_error(analyzed_func, expected):
    """Find error of an AnalyzedFunc object."""
    actual = analyzed_func.signed_area
    error = actual / expected
    return abs(1 - error)


def unsigned_area_error(analyzed_func, expected):
    """Find error of an AnalyzedFunc object."""
    actual = analyzed_func.unsigned_area
    error = actual / expected
    return abs(1 - error)


def test_trig_func_signed_area(analyzed_trig_func):
    """Test definite integral of trig_func is accurate."""
    error = signed_area_error(
        analyzed_trig_func, constants.TRIG_FUNC_SIGNED_AREA
    )
    assert error < 5.81425e-5


def test_trig_func_unsigned_area(analyzed_trig_func):
    """Test definite integral of trig_func is accurate."""
    error = unsigned_area_error(
        analyzed_trig_func, constants.TRIG_FUNC_UNSIGNED_AREA
    )
    assert error < 5.55566e-4
