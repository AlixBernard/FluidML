#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-03-24 16:11:27
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-08 02:24:06

""" Setup file. """

# Built-in packages
from setuptools import setup, find_packages

# Thid party packages

# Local packages


MAJOR = 0
MINOR = 9
PATCH = 0
ISRELEASED = False
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"


def get_version_info():
    FULLVERSION = VERSION
    GIT_REVISION = ""
    if not ISRELEASED:
        FULLVERSION += ".dev" + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename="fluidml/version.py"):
    cnt = """# THIS FILE IS GENERATED FROM FLUIDML SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    with open(filename, "w") as a:
        a.write(
            cnt
            % {
                "version": VERSION,
                "full_version": FULLVERSION,
                "git_revision": GIT_REVISION,
                "isrelease": str(ISRELEASED),
            }
        )


write_version_py()

with open("README.md") as file:
    long_description = file.read()

with open("requirements.txt", "r") as file:
    requirements = file.readlines()

setup(
    name="fluidml",
    version=VERSION,
    packages=find_packages(),
    author="Alix Bernard",
    author_email="alix.bernard9@gmail.com",
    description=(
        "Implementation of machine learning algorithms for fluid dynamics"
    ),
    long_description=str(long_description),
    license="MIT",
    install_requires=[req for req in requirements if not req.startswith("#")],
    url="https://github.com/NotAssigned",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Fluid Dynamics researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Fluid Dynamics",
    ],
)
