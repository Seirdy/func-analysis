# Changelog

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

[Full Changelog]

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

[Full Changelog]:
https://gitlab.com/Seirdy/func-analysis/compare/0.1.1...master

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
  - Predefined unit tests are in the submodoule
   `func_analysis.tests.test_all_analysis`. More testing submodules under
   `func_analysis.tests` will come soon.

### Fixed

- Corrected (and expanded) type annotations.
- The parameter `known_zeros` in `AnalyzedFunc.__init__()` is optional.

## [0.0.1] (2018-12-11)

[Full Changelog](https://gitlab.com/Seirdy/func-analysis/commits/0.0.1)

Initial release

[Unreleased]: https://gitlab.com/Seirdy/func-analysis/tree/master
[0.1.1]: https://gitlab.com/Seirdy/func-analysis/tree/0.1.1
[0.1.0]: https://gitlab.com/Seirdy/func-analysis/tree/0.1.0
[0.0.1]: https://gitlab.com/Seirdy/func-analysis/tree/0.0.1
