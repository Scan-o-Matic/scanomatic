#!/usr/bin/env python
"""Part of analysis work-flow that holds a grid arrays"""
import os

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman", ""]
__license__ = "GPL v3.0"
__version__ = "v2.2.0"
__maintainer__ = "Martin Zackrisson"
__status__ = "Development"

__branch = "dev"


def get_version():
    return __version__


def get_branch():
    return __branch


def get_location():
    return os.path.dirname(__file__)
