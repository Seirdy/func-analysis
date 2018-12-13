#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package func-analysis."""

import ast
import re
from os import path
from sys import version_info

from setuptools import setup

assert version_info >= (3, 7, 0), "func_analysis requires Python 3.7+"

CURRENT_DIR = path.dirname(__file__)


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

    Use gitlab pipelines to generate correct version.
    If this isn't a GitLab pipeline, then exract single-source version
    from func_analysis/__init__.py.
    """
    func_analysis_init = "func_analysis/__init__.py"
    _version_re = re.compile(r"__version__\s+=\s+(?P<version>.*)")
    with open(func_analysis_init, "r", encoding="utf8") as f:
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
        "v3 or later (AGPLv3+)",  # NOQA
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
)
