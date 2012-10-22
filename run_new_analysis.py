#!/usr/bin/env python
"""Scan-o-Matic"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
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
import threading

#
# THREADING
# 

gobject.threads_init()

#
# INTERNAL DEPENDENCIES
#

import src.controller_main as controller
import src.view_main as view
import src.model_main as model

#
# EXECUTION BEHAVIOUR
#

if __name__ == "__main__":

    m = model.load_app_model()
    w = view.Main_Window(model=m)
    c = controller.Controller(view=w, model=m)

    gtk.main()
