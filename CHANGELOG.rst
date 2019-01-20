=========
Changelog
=========

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.0.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

`Unreleased <https://gitlab.com/Seirdy/func-analysis/tree/master>`__
--------------------------------------------------------------------

`Full
Changelog <https://gitlab.com/Seirdy/func-analysis/compare/0.2.0...master>`__

Added
~~~~~

-  Added new analysis: signed area and unsigned area of the function
   across the x-range.

Fixed
~~~~~

-  Handle multiple bad args passed to AnalyzedFunc.func.
-  Interval-finding no longer raises a ``StopIteration`` exception.

Changed
~~~~~~~

-  Put all pairing logic in the functions ``make_pairs`` and
   ``make_intervals`` in the module ``func_analysis.interval_util``, and
   make both functions generators.
-  Improve class cohesion for ``AnalyzedFunc`` and its parents.

   -  Favor composition over inheritance in some cases.
   -  AnalyzedFunc now has 8 parents (down from 10).
   -  Cohesion still sucks; still have a long way to go!

Internal
~~~~~~~~

-  Started using
   `LGTM <https://lgtm.com/projects/g/Seirdy/func-analysis/>`__ for code
   quality checks.
-  Move once-used functions from ``util.py`` to the submodules that use
   them.
-  Replace ``util.py`` with ``custom_types.py`` and
   ``interval_util.py``.
-  Testing improvements:

   -  Fix divide-by-0 error in
      ``test_other_analysis.test_zeroth_derivative_is_itself``.
   -  Fix some static analyzers missing some directories.

`0.2.0 <https://gitlab.com/Seirdy/func-analysis/tree/0.2.0>`__ (2018-12-27)
---------------------------------------------------------------------------

`Full
Changelog <https://gitlab.com/Seirdy/func-analysis/compare/0.1.2...0.2.0>`__

Summary
~~~~~~~

This update contains many breaking changes.

With adoption of
`wemake-python-styleguide <https://wemake-python-styleguide.rtfd.io>`__
came many stylistic improvements. The massive analysis_classes.py file
has been split up, and much of the logic in ``__init__()`` methods now
resides in properties.

.. _fixed-1:

Fixed
~~~~~

-  ``zeros_wanted``, ``crits_wanted``, and ``pois_wanted`` cannot be
   negative anymore.

.. _added-1:

Added
~~~~~

-  ``AnalyzedFunc.func_real`` and ``AnalyzedFunc.func_iterable`` are
   limited versions of AnalyzedFunc.func.

.. _changed-1:

Changed
~~~~~~~

-  In ``AnalyzedFunc.__init__``, rename parameters ``known_zeros``,
   ``known_crits``, and ``known_pois`` to ``zeros``, ``crits``, and
   ``pois``.
-  Use more ``@property`` decorators.

   -  Convert some attrs to properties in ``AnalyzedFunc``:

      -  ``x_range``
      -  ``derivatives``

   -  Convert some methods to properties in ``AnalyzedFunc``. These no
      longer need empty double-parens:

      -  ``rooted_first_derivative`` and ``rooted_second_derivative``
      -  ``vertical_axis_of_symmetry``
      -  ``increasing`` and ``decreasing``
      -  ``concave`` and ``convex``
      -  ``relative_maxima`` and ``relative_minima``
      -  ``absolute_maximum`` and ``absolute_minimum``

-  The base version of ``AnalyzedFunc.func`` throws an error for
   unsupported types.
-  Better typing by subclassing ``NamedTuple``:

   -  Class ``Interval`` has fields ``start`` and ``stop``. It’s the
      return type of:

      -  ``x-range``
      -  ``increasing`` and ``decreasing``
      -  ``concave`` and ``convex``

   -  Class ``Coordinate`` has fields ``x_val`` and ``y_val``. It will
      be used more in a future update.

.. _internal-1:

Internal
~~~~~~~~

-  Split ``analysis_classes`` into ``af_base``, ``af_zeros``,
   ``af_crits_pois``, and ``analyzed_func``.
-  Prefer the stdlib version of @singledispatchmethod
-  Testing improvements

   -  Add tests to compare constructing AnalyzedFunc objects
      with/without known special points. Now at 100% test coverage!
   -  Splitting large modules

      -  Split ``tests.helper`` into ``tests.call_counting`` and
         ``tests.testing_utils``.
      -  Split ``test_zeros_crits_pois`` into ``test_zeros``,
         ``test_pois``, ``test_crits``
      -  Move extrema-testing from ``test_other_analysis`` to
         ``test_extrema``.
      -  Move all functions to analyze from ``conftest`` to
         ``funcs_to_analyze``.

   -  Linting: add ``wemake-python-styleguide`` and OpenStack’s
      ``hacking`` plugins to ``flake8``
   -  Rename long test methods
   -  Count-calling that existed only to ensure that a call-count never
      went past 0 has been replaced by tests that forbid calling
      altogether.

-  Minor changes

   -  Switch from ``os.path`` to ``pathlib.Path``.
   -  Switch from relative imports to absolute imports.
   -  Stop numeric underscore normalization
   -  Stop un-pythonic comparisons with zero
   -  Stop separating numerals from letters with underscores.
   -  Explicit object inheritance
   -  Spelling

.. _section-1:

`0.1.2 <https://gitlab.com/Seirdy/func-analysis/tree/0.1.2>`__ (2018-12-19)
---------------------------------------------------------------------------

`Full
Changelog <https://gitlab.com/Seirdy/func-analysis/compare/0.1.1...0.1.2>`__

.. _summary-1:

Summary
~~~~~~~

A bugfix in ``AnalyzedFunc`` and a ton of testing and pipeline
improvements.

.. _fixed-2:

Fixed
~~~~~

-  Special point properties work correctly when no special points are
   wanted.

.. _added-2:

Added
~~~~~

-  More tests

   -  Tests for ``AnalyzedFunc.concave`` and ``AnalyzedFunc.convex``
   -  Tests for memoization of ``AnalyzedFunc.func`` by monitoring call
      counts.

-  Pipeline additions

   -  Upload coverage to Code Climate
   -  Add xenon job to monitor code complexity

.. _changed-2:

Changed
~~~~~~~

-  Testing improvements

   -  Use fixtures to make all tests independent.
   -  Massive cleanup of ``testing.test_util``
   -  More files covered by linters (fixed glob patterns).
   -  Move helping functions and constants to ``tests.helpers`` and
      ``tests.constants``.
   -  Replace ``tests.test_all_analysis`` with
      ``tests.test_zeros_crits_pois``, ``tests.test_intervals``, and
      ``tests.test_other_analysis``.

-  More consistent formatting.

.. _section-2:

`0.1.1 <https://gitlab.com/Seirdy/func-analysis/tree/0.1.1>`__ (2018-12-17)
---------------------------------------------------------------------------

`Full
Changelog <https://gitlab.com/Seirdy/func-analysis/compare/0.1.0...0.1.1>`__

Hotfix release identical to 0.1.0 because I accidentally uploaded the
wrong file to PYPI.

.. _section-3:

`0.1.0 <https://gitlab.com/Seirdy/func-analysis/tree/0.1.0>`__ (2018-12-17)
---------------------------------------------------------------------------

`Full
Changelog <https://gitlab.com/Seirdy/func-analysis/compare/0.0.1...0.1.0>`__

.. _added-3:

Added
~~~~~

-  This changelog
-  Built-in test suite: ``python3 setup.py test`` runs unit tests
-  More badges to feed my badge addiction. More might come in the next
   version!

.. _changed-3:

Changed
~~~~~~~

-  Now there is only one public class for analyzed functions:
   ``AnalyzedFunc``. It has the same capabilities as ``FuncIntervals``
   from v0.0.1
-  ``AnalyzedFunc.zeros``, ``AnalyzedFunc.crits``, and
   ``AnalyzedFunc.pois`` are properties instead of ordinary methods;
   don’t use empty parentheses on these anymore!
-  Project structure

   -  ``func_analysis.func_analysis`` is now just ``func_analysis``.
   -  Predefined unit tests are in the submodule
      ``func_analysis.tests.test_all_analysis``. More testing submodules
      under ``func_analysis.tests`` will come soon.

.. _fixed-3:

Fixed
~~~~~

-  Corrected (and expanded) type annotations.
-  The parameter ``known_zeros`` in ``AnalyzedFunc.__init__()`` is
   optional.

.. _section-4:

`0.0.1 <https://gitlab.com/Seirdy/func-analysis/tree/0.0.1>`__ (2018-12-11)
---------------------------------------------------------------------------

`Full
Changelog <https://gitlab.com/Seirdy/func-analysis/commits/0.0.1>`__

Initial release
