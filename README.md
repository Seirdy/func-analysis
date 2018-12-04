# Function Analysis

[![pipeline status](https://gitlab.com/Seirdy/func-analysis/badges/master/pipeline.svg)](https://gitlab.com/Seirdy/func-analysis/commits/master)
[![coverage report](https://gitlab.com/Seirdy/func-analysis/badges/master/coverage.svg)](https://gitlab.com/Seirdy/func-analysis/commits/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This library uses concepts typically taught in an introductory Calculus class to describe properties of continuous, differentiable, single-variable functions.

# Using this library

`func_analysis.py` defines the class `FuncIntervals`. An instance of this class has several attributes describing the behavior of this function.

Required data include:

- A range
- The function to be analyzed

Special points include zeros, critical numbers, extrema, and points of inflection. Calculating these is possible when given the number of points wanted.

Optional data can be provided to improve precision and performance. Such data include:

- Any derivatives of the function
- Any known zeros, critical numbers, extrema, points of inflection
- Intervals of concavity, convexity, increase, decrease
- Any vertical axis of symmetry

# License

This program is licensed under the GNU Affero General Public License v3 or later.
