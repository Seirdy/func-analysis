=================
Function Analysis
=================

+------------------+-------------------------------------------------------+
| CI & Test Status | |gitlab-ci| |coverage|                                |
+------------------+-------------------------------------------------------+
| Code Quality     | |codeclimate| |codacy| |codebeat| |codefactor| |LGTM| |
+------------------+-------------------------------------------------------+
| Code Style       | |code style: black|                                   |
+------------------+-------------------------------------------------------+
| Dependencies     | |requires|                                            |
+------------------+-------------------------------------------------------+
| Usage            | |license|                                             |
+------------------+-------------------------------------------------------+
| PyPI             | |python version| |latest release|                     |
+------------------+-------------------------------------------------------+


This library uses concepts typically taught in an introductory Calculus
class to describe properties of continuous, differentiable,
single-variable functions.

Using this library
------------------

The ``func_analysis`` module defines the class ``AnalyzedFunc``. An
instance of this class has several attributes describing the behavior of
this function.

Required data include:

-  A range
-  The function to be analyzed

Special points include zeros, critical numbers, extrema, and points of
inflection. It’s possible to calculate these when given the number of
points wanted.

Optional data can be provided to improve precision and performance. Such
data include:

-  Any derivatives of the function
-  Any known zeros, critical numbers, extrema, points of inflection
-  Intervals of concavity, convexity, increase, decrease
-  Any vertical axis of symmetry

Any of the above data can be calculated by an instance of
``AnalyzedFunc``.

License
-------

This program is licensed under the GNU Affero General Public License v3
or later.

.. |gitlab-ci| image:: https://gitlab.com/Seirdy/func-analysis/badges/master/pipeline.svg
   :target: https://gitlab.com/Seirdy/func-analysis/commits/master
   :alt: Gitlab Pipeline Status
.. |coverage| image:: https://gitlab.com/Seirdy/func-analysis/badges/master/coverage.svg
   :target: https://gitlab.com/Seirdy/func-analysis/commits/master
   :alt: Coverage Report
.. |codeclimate| image:: https://codeclimate.com/github/Seirdy/func-analysis/badges/gpa.svg
   :target: https://codeclimate.com/github/Seirdy/func-analysis
   :alt: Code Climate
.. |codacy| image:: https://api.codacy.com/project/badge/Grade/cd4ff1fd5f26481f9da4e9f8a1ee8b7a
    :target: https://www.codacy.com/app/Seirdy/func-analysis
    :alt: Codacy
.. |codebeat| image:: https://codebeat.co/badges/439f2845-f06f-483c-848d-50633cae37bd
   :target: https://codebeat.co/projects/gitlab-com-seirdy-func-analysis-master
   :alt: codebeat badge
.. |codefactor| image:: https://www.codefactor.io/repository/github/seirdy/func-analysis/badge
   :target: https://www.codefactor.io/repository/github/seirdy/func-analysis
   :alt: CodeFactor
.. |LGTM| image:: https://img.shields.io/lgtm/alerts/g/Seirdy/func-analysis.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/Seirdy/func-analysis/
.. |license| image:: https://img.shields.io/pypi/l/func-analysis.svg
   :target: https://gitlab.com/Seirdy/func-analysis/blob/master/LICENSE
.. |python version| image:: https://img.shields.io/pypi/pyversions/func-analysis.svg?logo=python
   :target: https://pypi.org/project/func-analysis/
.. |latest release| image:: https://img.shields.io/pypi/v/func-analysis.svg
   :target: https://pypi.org/project/func-analysis/
.. |code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
.. |requires| image:: https://requires.io/enterprise/Seirdy/func-analysis/requirements.svg?branch=MASTER
   :target: https://requires.io/enterprise/Seirdy/func-analysis/requirements/?branch=MASTER
   :alt: Requirements Status
