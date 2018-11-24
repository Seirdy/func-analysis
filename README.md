# Function Analysis

This library uses concepts typically taught in an introductory Calculus class to describe properties of continuous, differentiable, single-variable functions.

# Using this library

`func_analysis.py` defines the class `FuncIntervals`. An instance of this class has several attributes describing the behavior of this function.

Required attributes include:

- A range
- The function to be analyzed

Optional attributes include:

- Any derivatives of the function
- The number of zeros, critical numbers, extrema, points of inflection
- Any known zeros, critical numbers, extrema, points of inflection
- Intervals of concavity, convexity, increase, decrease
- Any vertical axis of symmetry

An instance of `FuncIntervals` will contain *all* the above attributes, whether or not it was provided with them when instantiated. Providing any of the above during instantiation will improve the speed and/or accuracy of computation.

# License

This program is licensed under the GNU Affero General Public License v3 or later.
