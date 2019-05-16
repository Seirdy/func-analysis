# -*- coding: utf-8 -*-

"""Analyzer of the behavior of mathematical functions.

Given functions that are real, continuous, and differentiable
across specified intervals, this module graphs the function and
performs calculus to determine:

1. Special points
    - roots
    - critical numbers, relative/absolute extrema, and saddle points
    - points of inflection
2. Special intervals
    - intervals of increase/decrease
    - Intervals of concavity

Optional data can be provided to improve precision and performance.
These data can be:

- Any of the above data
- The number of any type of special point
- The first, second, and/or third derivatives of the function

This code uses the Black formatter.
"""

__version__ = "0.3.0"
from func_analysis.analyzed_func.analyzed_func import (  # noqa: F401
    AnalyzedFunc,
)
