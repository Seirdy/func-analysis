"""The classes that do the actual function analysis."""


from func_analysis.analyzed_func.af_base import AnalyzedFuncArea
from func_analysis.analyzed_func.af_intervals_extrema import (
    AnalyzedFuncExtrema,
    AnalyzedFuncIntervals,
)


class AnalyzedFunc(
    AnalyzedFuncIntervals, AnalyzedFuncExtrema, AnalyzedFuncArea
):
    """Complete function analysis, with special points and intervals."""
