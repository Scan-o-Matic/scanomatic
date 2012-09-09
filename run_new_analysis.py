#!/usr/bin/env python

import pygtk
pygtk.require('2.0')
import gtk

import src.controller_analysis as ca

w = gtk.Window()
c = ca.Analysis_Controller()
w.add(c.get_view())
w.show_all()

w.connect("delete_event", gtk.main_quit)
gtk.main()
