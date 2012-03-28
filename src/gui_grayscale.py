#!/usr/bin/env python
"""The GTK-GUI for showing the grayscale analysis"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.992"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import pygtk
pygtk.require('2.0')

import gtk, pango
import os, os.path, sys
import types

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# SCANNOMATIC LIBRARIES
#

import src.resource_image as img_base

#
# CLASSES
#


class Gray_Scale(gtk.Frame):

    def __init__(self, owner):

        label = "Grayscale analysis"
        gtk.Frame.__init__(self, label)

        self.owner = owner
        self.DMS = self.owner.DMS

        self.grayscale_fig = plt.Figure(figsize=(50,40), dpi=100)
        self.grayscale_fig.subplots_adjust(left=0.02, right=0.98, wspace=0.3)

        self.grayscale_plot_img = self.grayscale_fig.add_subplot(121)
        self.grayscale_plot_img.get_xaxis().set_visible(False)
        self.grayscale_plot_img.get_yaxis().set_visible(False)

        self.grayscale_plot = self.grayscale_fig.add_subplot(122)
        self.grayscale_plot.axis("tight")
        self.grayscale_plot.get_xaxis().set_visible(False)

        grayscale_canvas = FigureCanvas(self.grayscale_fig)
        grayscale_canvas.show()
        grayscale_canvas.set_size_request(400,150)

        self.add(grayscale_canvas)    
        self.show()

    def clf(self):

        self.grayscale_plot_img.clear()
        self.grayscale_plot.clear()

        self.grayscale_fig.canvas.draw()

    def set_grayscale(self, im_section):

        gs = img_base.Analyse_Grayscale(image=im_section)
        self._grayscale = gs._grayscale

        if gs._mid_orth_strip is None or gs._grayscale_pos is None:
            return False

        #LEFT PLOT
        Y = np.ones(len(gs._grayscale_pos)) * gs._mid_orth_strip 
        #grayscale_plot = self.grayscale_fig.get_subplot(121)
        self.grayscale_plot_img.clear()
        self.grayscale_plot_img.imshow(im_section.T)
        self.grayscale_plot_img.plot(gs._grayscale_pos, Y,'ko', mfc='w', mew=1, ms=3)
        self.grayscale_plot_img.set_xlim(xmin=0,xmax=im_section.shape[0])

        #RIGHT PLOT
        #grayscale_plot = self.grayscale_fig.get_subplot(122)

        if len(gs._grayscale_X) != len(gs._grayscale):
            self._grayscale=None
            self.DMS("Error", "There's something wrong with the grayscale. Switching to manual")
            return False

        z2 = np.polyfit(gs._grayscale_X, gs._grayscale,2)
        p2 = np.poly1d(z2)
        z3 = np.polyfit(gs._grayscale_X, gs._grayscale,3)
        p3 = np.poly1d(z3)

        xp = np.linspace(gs._grayscale_X[0], gs._grayscale_X[-1], 100)

        self.grayscale_plot.clear()
        line1 = self.grayscale_plot.plot(gs._grayscale_X, gs._grayscale,'b.', mfc='None', mew=2)
        #line2 = grayscale_plot.plot(xp, p2(xp),'r-')
        line3 = self.grayscale_plot.plot(xp, p3(xp),'g-')

        self.grayscale_fig.canvas.draw()

        return True
