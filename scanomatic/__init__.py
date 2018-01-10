#!/usr/bin/env python
"""The SoM package performs high accuracy and throughput phenotyping"""
import os

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman", ""]
__license__ = "GPL v3.0"
__version__ = "v3.0.0"
__maintainer__ = "Martin Zackrisson"
__status__ = "Development"

__branch = "dev"


def get_version():
    return __version__


def get_branch():
    return __branch


def get_location():
    return os.path.dirname(__file__)
