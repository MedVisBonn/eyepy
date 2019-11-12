#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "imageio==2.6.1",
    "numpy==1.17.3",
    "matplotlib==3.1.1, ",
    "scikit-image==0.16.2",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]

setup(
    author="Olivier Morelle",
    author_email="oli4morelle@gmail.com",
    python_requires=">2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="The Python package for working with ophthalmological data.",
    entry_points={"console_scripts": ["eyepy=eyepy.cli:main"]},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="eyepy",
    name="eyepy",
    packages=find_packages(include=["eyepy", "eyepy.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Oli4/eyepy",
    version="0.1.0",
    zip_safe=False,
)
