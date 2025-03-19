# -*- coding: utf-8 -*-
"""Setup script for this package."""

from setuptools import find_packages, setup

# we read in the required packages from the requirements.txt file
with open("requirements.txt", encoding="utf-8") as f:
    required_packages = f.read().splitlines()


setup(
    name="capu_time_series_analysis",
    version="0.0.1",
    url="https://github.com/ezeeeric/capu_time_series_analysis.git",
    author="Eric Drechsler",
    author_email="dr.eric.drechsler@gmail.com",
    description="Time Series Analysis Capilano University Analytics Team",
    packages=find_packages(),
    install_requires=required_packages,
    python_requires=">=3.11",
    include_package_data=True,
)
