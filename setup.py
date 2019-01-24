# -*- coding: utf-8 -*-

"""Package func-analysis."""

import re
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.test import test

PROJECT_ROOT = Path(__file__).parent


class PyTest(test):
    """Class to execute pytest as test suite without.

    To run tests, just type ``python3 setup.py test``.
    """

    def finalize_options(self):
        """Finalize pytest options."""
        test.finalize_options(self)
        self.test_args: list = []
        # defining attributes outside init is okay in
        # test.finalize_options().
        # pylint: disable = attribute-defined-outside-init
        self.test_suite = True
        # pylint: enable = attribute-defined-outside-init

    def run_tests(self):
        """Run pytest and exit with its exit code."""
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


def get_long_description() -> str:
    """Read README.rst.

    Returns
    -------
    long_description : str
        The text of README.rst.

    """
    with PROJECT_ROOT.joinpath("README.rst").open() as file_contents:
        return file_contents.read()


def get_version() -> str:
    """Determine correct version.

    Returns
    -------
    version : str
        The version found in ``func_analysis/__init__.py``.

    Raises
    ------
    RuntimeError
        If the string "__version__" cannot be found in the file
        ``func_analysis/__init__.py``.

    """
    version_path = PROJECT_ROOT.joinpath("func_analysis", "__init__.py")
    with version_path.open() as file_contents:
        version_file = file_contents.read()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


setup(
    name="func-analysis",
    version=get_version(),
    author="Rohan Kumar",
    author_email="seirdy@pm.ch",
    description="Analyze function behavior using introductory calculus.",
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    url="https://gitlab.com/Seirdy/func-analysis",
    packages=["func_analysis", "func_analysis.analyzed_func"],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License "
        + "v3 or later (AGPLv3+)",
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
