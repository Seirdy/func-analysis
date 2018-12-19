#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package func-analysis."""

import ast
import re
import sys
from os import path

from setuptools import setup
from setuptools.command.test import test

assert sys.version_info >= (3, 7, 0), "func_analysis requires Python 3.7+"

CURRENT_DIR = path.dirname(__file__)


class PyTest(test):
    """Class to execute pytest as test suite without.

    Also avoids the need to pre-install pytest on the system.
    To run tests, just type `python3 setup.py test`.
    """

    def finalize_options(self):
        """Finalize pytest options."""
        test.finalize_options(self)
        self.test_args: list = []
        # pylint: disable = attribute-defined-outside-init
        self.test_suite = True
        # pylint: enable = attribute-defined-outside-init

    def run_tests(self):
        """Run pytest and exit with its exit code."""
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


def get_long_description() -> str:
    """Read README.md.

    Returns
    -------
    str
        The text of README.md

    """
    with open(path.join(CURRENT_DIR, "README.md"), encoding="utf8") as fp:
        return fp.read()


def get_version() -> str:
    """Determine correct version.

    Use GitLab pipelines to generate correct version.
    If this isn't a GitLab pipeline, then extract single-source version
    from func_analysis/__init__.py.
    """
    func_analysis_init = "func_analysis/__init__.py"
    _version_re = re.compile(r"__version__\s+=\s+(?P<version>.*)")
    with open(func_analysis_init, encoding="utf8") as f:
        match = _version_re.search(f.read())
        version = match.group("version") if match is not None else '"unknown"'
    return str(ast.literal_eval(version))


setup(
    name="func-analysis",
    version=get_version(),
    author="Rohan Kumar",
    author_email="seirdy@pm.ch",
    description="Analyze function behavior using introductory calculus.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://gitlab.com/Seirdy/func-analysis",
    packages=["func_analysis"],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License "
        "v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    license="AGPLv3+",
    keywords=["func-analysis", "calculus", "math"],
    zip_safe=False,
    install_requires=["mpmath", "numpy", "scipy"],
    tests_require=["pytest>=4.0.1", "pytest-cov"],
    cmdclass={"pytest": PyTest},
)
