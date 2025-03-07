import glob
import re

from setuptools import setup, find_packages
from dsautils.version import get_git_version

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

#with open("burstfit/__init__.py", "r") as f:
#    vf = f.read()
#version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

version=get_git_version()

setup(
    name="burstfit",
    version=version,
    packages=find_packages(),
    url="https://github.com/thepetabyteproject/burstfit",
    author="Kshitij Aggarwal, Devansh Agarwal",
    scripts=glob.glob("bin/*"),
    tests_require=["pytest", "pytest-cov"],
    install_requires=required,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="ka0064@mix.wvu.edu, da0017@mix.wvu.edu",
    zip_safe=False,
    description="Spectro-temporal modeling of FRBs",
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
