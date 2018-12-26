# Changelog

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

[Full Changelog]

### Summary

This update contains many breaking changes.

With adoption of [wemake-python-styleguide] came many stylistic improvements.
The massive analysis_classes.py file has been split up, and much of the logic
in `__init__()` methods now resides in properties.

[wemake-python-styleguide]: https://wemake-python-styleguide.rtfd.io

### Added

- `AnalyzedFunc.func_real` and `AnalyzedFunc.func_iterable` are limited
  versions of AnalyzedFunc.func.

### Changed

- Use more `@property` decorators.
  - Convert some attrs to properties in `AnalyzedFunc`:
    - `x_range`
    - `derivatives`
  - Convert some methods to properties in `AnalyzedFunc`. These no longer need
    empty double-parens:
    - `rooted_first_derivative` and `rooted_second_derivative`
    - `vertical_axis_of_symmetry`
    - `increasing` and `decreasing`
    - `concave` and `convex`
    - `relative_maxima` and `relative_minima`
    - `absolute_maximum` and `absolute_minimum`
- The base version of `AnalyzedFunc.func` throws an error for unsupported
  types.
- Better typing by subclassing `NamedTuple`:
  - Class `Interval` has fields `start` and `stop`. It's the return type of:
    - `x-range`
    - `increasing` and `decreasing`
    - `concave` and `convex`
  - Class `Coordinate` has fields `x_val` and `y_val`. It will be used more in
    a future update.
- Split `analysis_classes` into `af_base`, `af_zeros`, `af_crits_pois`, and
  `analyzed_func`.
- Testing improvements
  - Add tests to compare constructing AnalyzedFunc objects with/without known
    special points. Now at 100% test coverage!
  - Splitting large modules
    - Split `tests.helper` into `tests.call_counting` and
      `tests.testing_utils`.
    - Split `test_zeros_crits_pois` into `test_zeros`, `test_pois`,
      `test_crits`
    - Move extrema-testing from `test_other_analysis` to `test_extrema`.
  - Move all functions to analyze from `conftest` to `funcs_to_analyze`.
  - Rename long test methods
  - Count-calling that existed only to ensure that a call-count never went
    past 0 has been replaced by tests that forbid calling altogether.
- Stylistic changes
  - Switch from relative imports to absolute imports.
  - Stop numeric underscore normalization
  - Stop un-pythonic comparisons with zero
  - Stop separating numerals from letters with underscores.
  - Explicit object inheritance
  - Spelling

[Full Changelog]:
https://gitlab.com/Seirdy/func-analysis/compare/0.1.2...master

## [0.1.2] (2018-12-19)

[Full Changelog](https://gitlab.com/Seirdy/func-analysis/compare/0.1.1...0.1.2)

### Summary

A bugfix in `AnalyzedFunc` and a ton of testing and pipeline improvements.

### Fixed

- Special point properties work correctly when no special points are wanted.

### Added

- More tests
  - Tests for `AnalyzedFunc.concave` and `AnalyzedFunc.convex`
  - Tests for memoization of `AnalyzedFunc.func` by monitoring call counts.
- Pipeline additions
  - Upload coverage to Code Climate
  - Add xenon job to monitor code complexity

### Changed

- Testing improvements
  - Use fixtures to make all tests independent.
  - Massive cleanup of `testing.test_util`
  - More files covered by linters (fixed glob patterns).
  - Move helping functions and constants to `tests.helpers` and
    `tests.constants`.
  - Replace `tests.test_all_analysis` with `tests.test_zeros_crits_pois`,
    `tests.test_intervals`, and `tests.test_other_analysis`.
- More consistent formatting.

## [0.1.1] (2018-12-17)

[Full Changelog](https://gitlab.com/Seirdy/func-analysis/compare/0.1.0...0.1.1)

Hotfix release identical to 0.1.0 because I accidentally uploaded the wrong
file to PYPI.

## [0.1.0] (2018-12-17)

[Full Changelog](https://gitlab.com/Seirdy/func-analysis/compare/0.0.1...0.1.0)

### Added

- This changelog
- Built-in test suite: `python3 setup.py test` runs unit tests
- More badges to feed my badge addiction. More might come in the next version!

### Changed

- Now there is only one public class for analyzed functions: `AnalyzedFunc`.
  It has the same capabilities as `FuncIntervals` from v0.0.1
- `AnalyzedFunc.zeros`, `AnalyzedFunc.crits`, and `AnalyzedFunc.pois` are
  properties instead of ordinary methods; don't use empty parentheses on these
  anymore!
- Project structure
  - `func_analysis.func_analysis` is now just `func_analysis`.
  - Predefined unit tests are in the submodule
   `func_analysis.tests.test_all_analysis`. More testing submodules under
   `func_analysis.tests` will come soon.

### Fixed

- Corrected (and expanded) type annotations.
- The parameter `known_zeros` in `AnalyzedFunc.__init__()` is optional.

## [0.0.1] (2018-12-11)

[Full Changelog](https://gitlab.com/Seirdy/func-analysis/commits/0.0.1)

Initial release

[Unreleased]: https://gitlab.com/Seirdy/func-analysis/tree/master
[0.1.2]: https://gitlab.com/Seirdy/func-analysis/tree/0.1.2
[0.1.1]: https://gitlab.com/Seirdy/func-analysis/tree/0.1.1
[0.1.0]: https://gitlab.com/Seirdy/func-analysis/tree/0.1.0
[0.0.1]: https://gitlab.com/Seirdy/func-analysis/tree/0.0.1
