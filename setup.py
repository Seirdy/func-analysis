#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package func-analysis."""

import sys
from os import path

from setuptools import find_packages, setup

assert sys.version_info >= (3, 7, 0), "func_analysis requires Python 3.7+"

CURRENT_DIR = path.dirname(__file__)


def read(*parts: str) -> str:
    """Read files in project directory.

    Parameters
    ----------
    *parts
        Relative paths in CURRENT_DIR of files to read.

    Returns
    -------
    str
        The text of all files in *paths.

    """
    with open(path.join(CURRENT_DIR, *parts), encoding="utf8") as fp:
        return fp.read()


# pylint: disable=line-too-long
setup(
    name="func-analysis",
    version="0.0.1",
    author="Rohan Kumar",
    author_email="seirdy@pm.ch",
    description="Analyze function behavior using introductory calculus.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://gitlab.com/Seirdy/func-analysis",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",  # NOQA
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    license="AGPLv3+",
    keywords=["func-analysis", "calculus", "math"],
    zip_safe=False,
    install_requires=["mpmath", "numpy", "scipy"],
)
