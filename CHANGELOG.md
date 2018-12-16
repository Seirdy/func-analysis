# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For more detailed changes, please see the [commit log](https://gitlab.com/Seirdy/func-analysis/commits/master).


## [Unreleased]
### Added
- This changelog.
- Built-in test suite: `python3 setup.py test` runs unit tests.
- More badges to feed my badge addiction.

### Changed
- Now there is only one public class for analyzed functions: `AnalyzedFunc`. It has the same capabilities as `FuncIntervals` from v0.0.1.
- `AnalyzedFunc.zeros`, `AnalyzedFunc.crits`, and `AnalyzedFunc.pois` are properties instead of ordinary methods. Don't use empty parentheses on these anymore,
- Project structure
	- `func_analysis.func_analysis` is now just `func_analysis`.
	- Predefined unit tests are in the submodoule `func_analysis.tests.test_all_analysis`.

## [0.0.1] - 2018-12-11
### Added
- Initial release

[Unreleased]: https://gitlab.com/Seirdy/func-analysis/compare/0.0.1...master
[0.0.1]: https://gitlab.com/Seirdy/func-analysis/commits/0.0.1
