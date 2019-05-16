=================
Function Analysis
=================

+------------------+-------------------------------------------------------+
| CI & Test Status | |Gitlab Pipeline Status| |Coverage Report|            |
+------------------+-------------------------------------------------------+
| Code Quality     | |Code Climate| |Codacy| |codebeat| |CodeFactor|       |
|                  | |LGTM|                                                |
+------------------+-------------------------------------------------------+
| Code Style       | |code style: black|                                   |
+------------------+-------------------------------------------------------+
| Dependencies     | |Requirements Status|                                 |
+------------------+-------------------------------------------------------+
| Usage            | |license|                                             |
+------------------+-------------------------------------------------------+
| PyPI             | |python version| |latest release|                     |
+------------------+-------------------------------------------------------+

This library uses concepts typically taught in an introductory Calculus class
to describe properties of continuous, differentiable, single-variable
functions.

Using this library
==================

The ``func_analysis`` module defines the class ``AnalyzedFunc``. An instance of
this class has several attributes describing the behavior of this function.

Required data include:

-  A range
-  The function to be analyzed

Special points include zeros, critical numbers, extrema, and points of
inflection. It’s possible to calculate these when given the number of points
wanted.

Optional data can be provided to improve precision and performance. Such data
include:

-  Any derivatives of the function
-  Any known zeros, critical numbers, extrema, points of inflection
-  Intervals of concavity, convexity, increase, decrease
-  Any vertical axis of symmetry

Any of the above data can be calculated by an instance of ``AnalyzedFunc``.

Example Usage
-------------

This paste from an interactive Python session showcases all the functionality
of ``AnalyzedFunc``:

.. code:: python

   >>> from func_analysis import AnalyzedFunc

   >>> import mpmath as mp; import numpy as np

   >>> mp.pretty = True

   >>> def example_func(x):
   ...     return mp.cos(x ** 2) - mp.sin(x) + (x / 68)
   ...

   >>> analyzed_example = AnalyzedFunc(
   ...     func=example_func,
   ...     x_range=(-47.05, -46.3499),
   ...     zeros_wanted=21,
   ...     crits_wanted=21,
   ...     pois_wanted=21,
   ...     zeros=[-47.038289673236127, -46.406755885040056],
   ... )

   >>> analyzed_example.zeros
   array([mpf('-47.038289673236127'), mpf('-47.018473233395284'),
          mpf('-46.972318087653945'), mpf('-46.950739626397913'),
          mpf('-46.906204518117636'), mpf('-46.882958270910017'),
          mpf('-46.839955720658347'), mpf('-46.815121707485004'),
          mpf('-46.77357601136889'), mpf('-46.747224922729004'),
          mpf('-46.707068062964038'), mpf('-46.679264553080846'),
          mpf('-46.640433373296687'), mpf('-46.611238416225623'),
          mpf('-46.57367255467036'), mpf('-46.543145221101676'),
          mpf('-46.506785519620839'), mpf('-46.474984380574834'),
          mpf('-46.439771604599501'), mpf('-46.406755885040056'),
          mpf('-46.372629655875102')], dtype=object)

   >>> analyzed_example.crits
   array([mpf('-47.028400867252276'), mpf('-46.995216177440788'),
          mpf('-46.961552135996999'), mpf('-46.928318300227147'),
          mpf('-46.894608617023608'), mpf('-46.861324416365338'),
          mpf('-46.827569901478539'), mpf('-46.794234116960419'),
          mpf('-46.760435575283916'), mpf('-46.72704699248236'),
          mpf('-46.693205219083057'), mpf('-46.659762632756908'),
          mpf('-46.625878408195688'), mpf('-46.592380626945829'),
          mpf('-46.558454712583136'), mpf('-46.524900563516225'),
          mpf('-46.490933696823733'), mpf('-46.457322030198712'),
          mpf('-46.42331492009863'), mpf('-46.389644613934263'),
          mpf('-46.355597936188148')], dtype=object)

   >>> analyzed_example.pois
   array([mpf('-47.04521505151731'), mpf('-47.011813891641338'),
          mpf('-46.978389522478297'), mpf('-46.944940655832212'),
          mpf('-46.911468800195282'), mpf('-46.877972023301726'),
          mpf('-46.844452476333207'), mpf('-46.81090758498618'),
          mpf('-46.777340139630388'), mpf('-46.743746928888186'),
          mpf('-46.710131375870707'), mpf('-46.676489640045468'),
          mpf('-46.64282576785548'), mpf('-46.60913530049924'),
          mpf('-46.575422895374974'), mpf('-46.541683489262155'),
          mpf('-46.507922335179564'), mpf('-46.474133782285819'),
          mpf('-46.440323660950513'), mpf('-46.406485752427894'),
          mpf('-46.372626443270374')], dtype=object)

   >>> analyzed_example.increasing
   [Interval(start=-47.05, stop=mpf('-47.028400867252276')),
    Interval(start=mpf('-46.995216177440788'), stop=mpf('-46.961552135996999')),
    Interval(start=mpf('-46.928318300227147'), stop=mpf('-46.894608617023608')),
    Interval(start=mpf('-46.861324416365338'), stop=mpf('-46.827569901478539')),
    Interval(start=mpf('-46.794234116960419'), stop=mpf('-46.760435575283916')),
    Interval(start=mpf('-46.72704699248236'), stop=mpf('-46.693205219083057')),
    Interval(start=mpf('-46.659762632756908'), stop=mpf('-46.625878408195688')),
    Interval(start=mpf('-46.592380626945829'), stop=mpf('-46.558454712583136')),
    Interval(start=mpf('-46.524900563516225'), stop=mpf('-46.490933696823733')),
    Interval(start=mpf('-46.457322030198712'), stop=mpf('-46.42331492009863')),
    Interval(start=mpf('-46.389644613934263'), stop=mpf('-46.355597936188148'))]

   >>> analyzed_example.decreasing
   [Interval(start=mpf('-47.028400867252276'), stop=mpf('-46.995216177440788')),
    Interval(start=mpf('-46.961552135996999'), stop=mpf('-46.928318300227147')),
    Interval(start=mpf('-46.894608617023608'), stop=mpf('-46.861324416365338')),
    Interval(start=mpf('-46.827569901478539'), stop=mpf('-46.794234116960419')),
    Interval(start=mpf('-46.760435575283916'), stop=mpf('-46.72704699248236')),
    Interval(start=mpf('-46.693205219083057'), stop=mpf('-46.659762632756908')),
    Interval(start=mpf('-46.625878408195688'), stop=mpf('-46.592380626945829')),
    Interval(start=mpf('-46.558454712583136'), stop=mpf('-46.524900563516225')),
    Interval(start=mpf('-46.490933696823733'), stop=mpf('-46.457322030198712')),
    Interval(start=mpf('-46.42331492009863'), stop=mpf('-46.389644613934263')),
    Interval(start=mpf('-46.355597936188148'), stop=-46.3499)]

   >>> analyzed_example.concave
   [Interval(start=-47.05, stop=mpf('-47.04521505151731')),
    Interval(start=mpf('-47.011813891641338'), stop=mpf('-46.978389522478297')),
    Interval(start=mpf('-46.944940655832212'), stop=mpf('-46.911468800195282')),
    Interval(start=mpf('-46.877972023301726'), stop=mpf('-46.844452476333207')),
    Interval(start=mpf('-46.81090758498618'), stop=mpf('-46.777340139630388')),
    Interval(start=mpf('-46.743746928888186'), stop=mpf('-46.710131375870707')),
    Interval(start=mpf('-46.676489640045468'), stop=mpf('-46.64282576785548')),
    Interval(start=mpf('-46.60913530049924'), stop=mpf('-46.575422895374974')),
    Interval(start=mpf('-46.541683489262155'), stop=mpf('-46.507922335179564')),
    Interval(start=mpf('-46.474133782285819'), stop=mpf('-46.440323660950513')),
    Interval(start=mpf('-46.406485752427894'), stop=mpf('-46.372626443270374'))]

   >>> analyzed_example.convex
   [Interval(start=mpf('-47.04521505151731'), stop=mpf('-47.011813891641338')),
    Interval(start=mpf('-46.978389522478297'), stop=mpf('-46.944940655832212')),
    Interval(start=mpf('-46.911468800195282'), stop=mpf('-46.877972023301726')),
    Interval(start=mpf('-46.844452476333207'), stop=mpf('-46.81090758498618')),
    Interval(start=mpf('-46.777340139630388'), stop=mpf('-46.743746928888186')),
    Interval(start=mpf('-46.710131375870707'), stop=mpf('-46.676489640045468')),
    Interval(start=mpf('-46.64282576785548'), stop=mpf('-46.60913530049924')),
    Interval(start=mpf('-46.575422895374974'), stop=mpf('-46.541683489262155')),
    Interval(start=mpf('-46.507922335179564'), stop=mpf('-46.474133782285819')),
    Interval(start=mpf('-46.440323660950513'), stop=mpf('-46.406485752427894')),
    Interval(start=mpf('-46.372626443270374'), stop=-46.3499)]

   >>> analyzed_example.relative_maxima
   array([mpf('-47.028400867252276'), mpf('-46.961552135996999'),
          mpf('-46.894608617023608'), mpf('-46.827569901478539'),
          mpf('-46.760435575283916'), mpf('-46.693205219083057'),
          mpf('-46.625878408195688'), mpf('-46.558454712583136'),
          mpf('-46.490933696823733'), mpf('-46.42331492009863'),
          mpf('-46.355597936188148')], dtype=object)

   >>> analyzed_example.relative_minima
   array([mpf('-46.995216177440788'), mpf('-46.928318300227147'),
          mpf('-46.861324416365338'), mpf('-46.794234116960419'),
          mpf('-46.72704699248236'), mpf('-46.659762632756908'),
          mpf('-46.592380626945829'), mpf('-46.524900563516225'),
          mpf('-46.457322030198712'), mpf('-46.389644613934263')],
         dtype=object)

   >>> analyzed_example.absolute_maximum
   Coordinate(x_val=mpf('-46.355597936188148'), y_val=mpf('1.0131766438615282'))

   >>> analyzed_example.absolute_minimum
   Coordinate(x_val=mpf('-46.995216177440788'), y_val=mpf('-1.5627299417380764'))

   >>> analyzed_example.signed_area
   mpf('-0.1835790011406907')

   >>> analyzed_example.unsigned_area
   mpf('0.46577475660746492')

We can see that the inflection points of a function, the critical points of its
first derivative, and the zeros of its second derivative are identical.

.. code:: python

   >>> np.array_equal(
   ...     analyzed_example.pois, analyzed_example.rooted_first_derivative.crits
   ... )
   True

   >>> np.array_equal(
   ...     analyzed_example.pois, analyzed_example.rooted_second_derivative.zeros
   ... )
   True

Other examples to demonstrate the relationship between derivatives:

.. code:: python

   >>> np.array_equal(analyzed_example.concave, analyzed_example.rooted_first_derivative.increasing)
   True

   >>> np.array_equal(analyzed_example.first_derivative.convex, analyzed_example.rooted_second_derivative.decreasing)
   True

A work-in-progress feature is listing x-values of vertical axes of symmetry.
Here's an example of a function that's symmetric about the y-axis:

.. code:: python

   >>> def symmetric_func(x):
           return mp.power(x, 2) - 4

   >>> analyzed_symmetric_example = AnalyzedFunc(
   ...     func=lambda x: mp.power(x, 2) - 4,
   ...     x_range=(-8,8),
   ...     zeros_wanted=2
   ... )

   >>> analyzed_symmetric_example.vertical_axis_of_symmetry
   [0.0]

License
=======

This program is licensed under the GNU Affero General Public License v3 or
later.

.. |Gitlab Pipeline Status| image:: https://gitlab.com/Seirdy/func-analysis/badges/master/pipeline.svg
   :target: https://gitlab.com/Seirdy/func-analysis/commits/master
.. |Coverage Report| image:: https://gitlab.com/Seirdy/func-analysis/badges/master/coverage.svg
   :target: https://gitlab.com/Seirdy/func-analysis/commits/master
.. |Code Climate| image:: https://codeclimate.com/github/Seirdy/func-analysis/badges/gpa.svg
   :target: https://codeclimate.com/github/Seirdy/func-analysis
.. |Codacy| image:: https://api.codacy.com/project/badge/Grade/cd4ff1fd5f26481f9da4e9f8a1ee8b7a
   :target: https://www.codacy.com/app/Seirdy/func-analysis
.. |codebeat| image:: https://codebeat.co/badges/439f2845-f06f-483c-848d-50633cae37bd
   :target: https://codebeat.co/projects/gitlab-com-seirdy-func-analysis-master
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/seirdy/func-analysis/badge
   :target: https://www.codefactor.io/repository/github/seirdy/func-analysis
.. |LGTM| image:: https://img.shields.io/lgtm/alerts/g/Seirdy/func-analysis.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/Seirdy/func-analysis/
.. |code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
.. |Requirements Status| image:: https://requires.io/enterprise/Seirdy/func-analysis/requirements.svg?branch=MASTER
   :target: https://requires.io/enterprise/Seirdy/func-analysis/requirements/?branch=MASTER
.. |license| image:: https://img.shields.io/pypi/l/func-analysis.svg
   :target: https://gitlab.com/Seirdy/func-analysis/blob/master/LICENSE
.. |python version| image:: https://img.shields.io/pypi/pyversions/func-analysis.svg?logo=python
   :target: https://pypi.org/project/func-analysis/
.. |latest release| image:: https://img.shields.io/pypi/v/func-analysis.svg
   :target: https://pypi.org/project/func-analysis/
