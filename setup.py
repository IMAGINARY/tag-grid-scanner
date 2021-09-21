# -*- coding: utf-8 -*-


"""setup.py: setuptools control."""


import re
from setuptools import setup

from taggridscanner.version import string as version_string

with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")


setup(
    name="cmdline-taggridscanner",
    packages=["bootstrap"],
    entry_points={
        "console_scripts": ["tag-grid-scanner = taggridscanner.taggridscanner:main"]
    },
    version=version_string,
    description="Scan (a stream of) images for a grid of tags",
    long_description=long_description,
    author="Christian Stussak",
    author_email="christian.stussak@imaginary.org",
    url="https://github.com/IMAGINARY/tag-grid-scanner",
)
