#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package func-analysis."""

import ast
import re
import sys
from os import environ, path

from setuptools import find_packages, setup

assert sys.version_info >= (3, 7, 0), "func_analysis requires Python 3.7+"

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
    from func_analysis/func_analysis.py.
    """
    if environ.get("CI_COMMIT_TAG"):
        return environ["CI_COMMIT_TAG"]
    if environ.get("CI_JOB_ID"):
        return environ["CI_JOB_ID"]
    func_analysis_py = "func_analysis/func_analysis.py"
    _version_re = re.compile(r"__version__\s+=\s+(?P<version>.*)")
    with open(func_analysis_py, "r", encoding="utf8") as f:
        match = _version_re.search(f.read())
        version = match.group("version") if match is not None else '"unknown"'
    return str(ast.literal_eval(version))


# pylint: disable=line-too-long
setup(
    name="func-analysis",
    version=get_version(),
    author="Rohan Kumar",
    author_email="seirdy@pm.ch",
    description="Analyze function behavior using introductory calculus.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://gitlab.com/Seirdy/func-analysis",
    packages=find_packages(),
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
