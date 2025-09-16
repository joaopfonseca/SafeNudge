import os
from pathlib import Path
from setuptools import find_packages, setup


def get_long_description() -> str:
    CURRENT_DIR = Path(__file__).parent
    return (CURRENT_DIR / "README.md").read_text(encoding="utf8")


ver_file = os.path.join("ctg", "_version.py")
with open(ver_file) as f:
    exec(f.read())

MAINTAINER = "J. Fonseca"
MAINTAINER_EMAIL = "jpm9748@nyu.edu"
URL = "https://github.com/joaopfonseca/safenudge"
VERSION = __version__
SHORT_DESCRIPTION = (
    "Implementation of Machine Learning algorithms, experiments and utilities."
)
LICENSE = "MIT"
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
INSTALL_REQUIRES = [
    dep.strip() for dep in open("requirements.txt", "r").readlines() if dep.strip()
]

setup(
    name="ctg",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    download_url=URL,
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)
