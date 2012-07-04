#!/usr/bin/env python
"""The GTK-GUI for showing the grayscale analysis"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.996"
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
        self.grayscale_plot_img_ax = None
        self.grayscale_plot_img_ax2 = None

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

    def get_has_grayscale(self):

        if self._grayscale is None:
            return False

        else:
            return True

    def set_grayscale(self, im_section, scale_factor=1.0, dpi=600):

        if im_section is None:
            self._grayscale = None
            if self.grayscale_plot_img_ax is not None:
                self.grayscale_plot_img_ax.set_visible(False)
                self.grayscale_plot_img_ax2.set_visible(False)

            self.DMS('GRAYSCALE', 'Gray-scale cleared since no image passed', 
                "L", debug_level='info')

            return False

        gs = img_base.Analyse_Grayscale(image=im_section, scale_factor=scale_factor, dpi=dpi)
        self._grayscale = gs._grayscale

        if gs._mid_orth_strip is None or gs._grayscale_pos is None:
            self.DMS('GRAYSCALE', 'Gray-scale could not find any signal.', 
                "DL", debug_level='error')

            if self.grayscale_plot_img_ax is not None:
                self.grayscale_plot_img_ax.set_visible(False)
                self.grayscale_plot_img_ax2.set_visible(False)
            return False

        #LEFT PLOT
        Y = np.ones(len(gs._grayscale_pos)) * gs._mid_orth_strip 
        #grayscale_plot = self.grayscale_fig.get_subplot(121)
        #self.grayscale_plot_img.clear()
        #if self.grayscale_plot_img_ax is None:
        self.grayscale_plot_img.cla()
        self.grayscale_plot_img_ax = self.grayscale_plot_img.imshow(\
            im_section.T, cmap=plt.cm.gray)
        self.grayscale_plot_img_ax2 = self.grayscale_plot_img.plot(\
            gs._grayscale_pos, Y,'o', mfc='r', mew=1, ms=3)[0]
        self.grayscale_plot_img.plot(gs._gray_scale_pos, 
            np.ones(len(gs._gray_scale_pos))*gs._mid_orth_strip - gs.ortho_half_height ,
            'r-' )
        self.grayscale_plot_img.plot(gs._gray_scale_pos, 
            np.ones(len(gs._gray_scale_pos))*gs._mid_orth_strip + gs.ortho_half_height,
            'r-' )
        #else:
            #self.grayscale_plot_img_ax.set_data(im_section.T)
            #self.grayscale_plot_img_ax2.set_xdata(gs._grayscale_pos)
            #self.grayscale_plot_img_ax2.set_ydata(Y)

        self.grayscale_plot_img.set_xlim(xmin=0,xmax=im_section.shape[0])
        self.grayscale_plot_img_ax.set_visible(True)
        self.grayscale_plot_img_ax2.set_visible(True)

        #RIGHT PLOT
        #grayscale_plot = self.grayscale_fig.get_subplot(122)
        kodak_values = gs.get_kodak_values()
        if len(kodak_values) != len(gs._grayscale):
            self._grayscale=None
            self.DMS("GRAYSCALE GUI", 
                "There's something wrong with the grayscale. Try manual selection",
                "DL", debug_level="error")
            return False

        #z2 = np.polyfit(kodak_values, gs._grayscale,2)
        #p2 = np.poly1d(z2)
        z3 = np.polyfit(kodak_values, gs._grayscale,3)
        p3 = np.poly1d(z3)

        if z3 is not None:


            z3_deriv_coeffs = np.array(z3[:-1]) * np.arange(z3.shape[0]-1,0,-1)

            z3_deriv = np.array(map(lambda x: (z3_deriv_coeffs*np.power(x,
                np.arange(z3_deriv_coeffs.shape[0],0,-1))).sum(), range(87)))

            if (z3_deriv > 0).any() and (z3_deriv < 0).any():

                self.DMS('GRAYSCALE GUI', 
                    'Grayscale is dubious since coefficients ({0}) don\'t\
 agree on sign.'.format(z3_deriv_coeffs),
                    level="DL", debug_level='error')

                self._grayscale=None
                return False
            else:
                self.DMS('GRAYSCALE GUI', 'Got regression coeffs: {0}'.format(z3), level="L",
                    debug_level='info')
        else:

            self.DMS('GRAYSCALE GUI',
                "Could not find a polynomial fitting the points",
                level="L", debug_level="error")
            self._grayscale=None
            return False

        xp = np.linspace(kodak_values[0], kodak_values[-1], 100)

        self.grayscale_plot.clear()
        line1 = self.grayscale_plot.plot(kodak_values, gs._grayscale,'bo', mfc='None', mew=1)
        #line2 = grayscale_plot.plot(xp, p2(xp),'r-')
        line3 = self.grayscale_plot.plot(xp, p3(xp),'g-', lw=2)

        self.grayscale_fig.canvas.draw()

        return True
