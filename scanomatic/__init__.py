#!/usr/bin/env python
"""Part of analysis work-flow that holds a grid arrays"""
import requests
from scanomatic.io.logger import Logger

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "v1.3.1"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

_logger = Logger("Scan-o-Matic")


def _version_parser(version=__version__):

    return tuple(int("".join(c for c in v if c in "0123456789")) for v in version.split(".")
                 if any((c in "0123456789" and c) for c in v))


def _git_version(
        git_repo='https://raw.githubusercontent.com/local-minimum/scanomatic',
        branch='master',
        file='scanomatic/__init__.py'):


    uri = "/".join((git_repo, branch, file))
    for line in requests.get(uri).text.split("\n"):
        if line.startswith("__version__"):
            return _version_parser(line.split("=")[-1].strip())

    _logger.warning("Could not access any valid version information from uri {0}".format(uri))
    return _version_parser("")


def _greatest_version(v1, v2):

    comparable = min(len(v) for v in (v1, v2))
    for i in range(comparable):
        if v1[i] == v2[i]:
            continue
        elif v1[i] > v2[i]:
            return v1
        else:
            return v2

    if len(v1) >= len(v2):
        return v1
    elif len(v2) > len(v1):
        return v2

    _logger.warning("None of the versions is a version!")
    return None


def is_newest_version(branch='master'):

    current = _version_parser()
    git_version = _git_version(branch=branch)
    if current == _greatest_version(current, git_version):
        _logger.info("Already using most recent version {0}".format(current))
        return True
    else:
        _logger.info("There's a new version on the branch {0} available.".format(branch))
        return False