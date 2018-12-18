# Function Analysis

[![pipeline status]](https://gitlab.com/Seirdy/func-analysis/commits/master)
[![coverage report]](https://gitlab.com/Seirdy/func-analysis/commits/master)
[![Code Climate]](https://codeclimate.com/github/Seirdy/func-analysis)
[![License]](https://gitlab.com/Seirdy/func-analysis/blob/master/LICENSE)
[![PYPI latest release]](https://pypi.org/project/func-analysis/)
[![Python version]](https://pypi.org/project/func-analysis/)
[![Code style: black]](https://github.com/ambv/black)

[pipeline status]:
https://gitlab.com/Seirdy/func-analysis/badges/master/pipeline.svg
[coverage report]:
https://gitlab.com/Seirdy/func-analysis/badges/master/coverage.svg
[Code Climate]:
https://codeclimate.com/github/Seirdy/func-analysis/badges/gpa.svg
[License]:
https://img.shields.io/pypi/l/func-analysis.svg
[PYPI Latest Release]:
https://img.shields.io/pypi/v/func-analysis.svg
[Python version]:
https://img.shields.io/pypi/pyversions/func-analysis.svg
[Code style: black]:
https://img.shields.io/badge/code%20style-black-000000.svg

This library uses concepts typically taught in an introductory Calculus
class to describe properties of continuous, differentiable, single-variable
functions.

## Using this library

The `func_analysis` module defines the class `AnalyzedFunc`. An instance
of this class has several attributes describing the behavior of this
function.

Required data include:

- A range
- The function to be analyzed

Special points include zeros, critical numbers, extrema, and points of
inflection. Calculating these is possible when given the number of points
wanted.

Optional data can be provided to improve precision and performance. Such
data include:

- Any derivatives of the function
- Any known zeros, critical numbers, extrema, points of inflection
- Intervals of concavity, convexity, increase, decrease
- Any vertical axis of symmetry

Any of the above data can be calculated by an instance of `AnalyzedFunc`.

## License

This program is licensed under the GNU Affero General Public License v3 or
later.
