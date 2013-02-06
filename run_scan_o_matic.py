#!/usr/bin/env python
"""Scan-o-Matic"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
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
from multiprocessing.pool import ThreadPool

#
# THREADING
# 

gobject.threads_init()

#
# INTERNAL DEPENDENCIES
#

import src.controller_main as controller
import src.view_main as view_main
import src.model_main as model_main

#
# EXECUTION BEHAVIOUR
#

if __name__ == "__main__":

    program_path = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != program_path:
        os.chdir(program_path)

    gobject.threads_init()
    splash = view_main.Splash(program_path)

    pool = ThreadPool(processes=1)
    async_result = pool.apply_async(controller.Controller,
        (program_path,))
    
    while gtk.events_pending():
       gtk.main_iteration() 

    c = async_result.get()

    splash.main_is_loaded()

    gtk.main()
