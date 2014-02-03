#!/usr/bin/env python
"""Scan-o-Matic"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import pygtk
pygtk.require('2.0')
import gtk
import gobject
import os
from argparse import ArgumentParser

#
# INTERNAL DEPENDENCIES
#

import src.gui.controller_main as controller

#
# EXECUTION BEHAVIOUR
#

if __name__ == "__main__":

    program_path = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != program_path:
        os.chdir(program_path)

    gobject.threads_init()

    parser = ArgumentParser(description="""The Scan-o-Matic GUI""")

    parser.add_argument("-d", "--debug", dest="debug", type=bool,
                        default=False, help="Run in debug mode")

    args = parser.parse_args()

    controller.Controller(program_path, debug_mode=args.debug)

    gtk.main()
