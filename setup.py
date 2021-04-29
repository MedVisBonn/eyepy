#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import sys

from setuptools import find_packages, setup

# These are also specified in eyepy.__init__.py
__author__ = """Olivier Morelle"""
__email__ = "oli4morelle@gmail.com"
__version__ = "0.3.0"

try:
    from semantic_release import setup_hook

    setup_hook(sys.argv)
except ImportError:
    pass

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["imageio", "numpy", "matplotlib", "seaborn", "scikit-image",
                "scipy", "imagecodecs"]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]

setup(
    author=__author__,
    author_email=__email__,
    python_requires=">=3.6",
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
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["eyepy", "eyepie"],
    name="eyepie",
    packages=find_packages(include=["eyepy", "eyepy.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/MedVisBonn/eyepy",
    version=__version__,
    zip_safe=False,
)
