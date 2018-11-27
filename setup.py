#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package func-analysis"""

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# pylint: disable=line-too-long
setup(
    name="func-analysis",
    version="0.0.0",
    author="Rohan Kumar",
    author_email="seirdy@pm.ch",
    description="Analyze function behavior using introductory calculus.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/Seirdy/func-analysis",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)", # NOQA
        "Topic :: Utilities",
    ],
)
