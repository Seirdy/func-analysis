# Changelog

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

[Full Changelog](https://gitlab.com/Seirdy/func-analysis/compare/0.0.1...master)

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
[0.0.1]: https://gitlab.com/Seirdy/func-analysis/commits/0.0.1
