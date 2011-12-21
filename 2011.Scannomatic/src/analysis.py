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
import matplotlib.patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
#import subprocess #To be removed

#
# SCANNOMATIC LIBRARIES
#

import src.image_analysis_base as img_base
import src.settings_tools as settings_tools
import src.colonies_wrapper as colonies

class Analyse_One(gtk.Frame):

    def __init__(self, owner, label="Analyse One Image"):

        gtk.Frame.__init__(self, label)

        self.owner = owner
        self.analysis = None

        self._rect_marking = False
        self._rect_ul = None
        self._rect_lr = None

        self._fixture_config_root = self.owner._program_config_root + os.sep + "fixtures"
        self.f_settings = settings_tools.Fixture_Settings(self._fixture_config_root, fixture="fixture_a")

        vbox = gtk.VBox()
        vbox.show()
        self.add(vbox)

        label = gtk.Label("THIS IS EXTREMELY PROTOTYPE DEVish")
        label.show()
        vbox.pack_start(label, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox.pack_start(hbox, False, False, 2)

        self.plots_vbox = gtk.VBox()
        self.plots_vbox.show()
        hbox.pack_start(self.plots_vbox, False, False, 2)
        

        self.plots_vbox2 = gtk.VBox()
        self.plots_vbox2.show()
        hbox.pack_start(self.plots_vbox2, False, False, 2)

        vbox2 = gtk.VBox()
        vbox2.show()
        hbox.pack_end(vbox2, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)

        label = gtk.Label("Select image:")
        label.show()
        hbox.pack_start(label, False, False, 2)

        button = gtk.Button(label = 'Open')
        button.show()
        button.connect("clicked", self.select_image)
        hbox.pack_end(button, False, False, 2)

        self.analysis_img = gtk.Label("")
        self.analysis_img.set_max_width_chars(40)
        self.analysis_img.set_ellipsize(pango.ELLIPSIZE_START)
        self.analysis_img.show()
        vbox2.pack_start(self.analysis_img, False, False, 2)

        #Analysis data frame for selection
        frame = gtk.Frame("Selection Analysis")
        frame.show()
        vbox2.pack_start(frame, False, False, 2)

        vbox3 = gtk.VBox()
        vbox3.show()
        frame.add(vbox3)

        #Interactive helper
        self.section_picking = gtk.Label("First load an image.")
        self.section_picking.show()
        vbox3.pack_start(self.section_picking, False, False, 10)

        #Cell Area
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Cell Area:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.cell_area = gtk.Label("0")
        self.cell_area.show()
        self.cell_area.set_max_width_chars(20)
        hbox.pack_end(self.cell_area, False, False, 2)

        #Background Mean
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Background Mean:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.bg_mean = gtk.Label("0")
        self.bg_mean.show()
        hbox.pack_end(self.bg_mean, False, False, 2)

        #Background Inter Quartile Range Mean
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Background IQR-Mean:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.bg_iqr_mean = gtk.Label("0")
        self.bg_iqr_mean.show()
        hbox.pack_end(self.bg_iqr_mean, False, False, 2)

        #Background Median
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Background Median:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.bg_median = gtk.Label("0")
        self.bg_median.show()
        hbox.pack_end(self.bg_median, False, False, 2)

        #Colony Size
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Colony Size:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.colony_size = gtk.Label("0")
        self.colony_size.show()
        hbox.pack_end(self.colony_size, False, False, 2)

    def select_image(self, widget=None, event=None, data=None):
        newimg = gtk.FileChooserDialog(title="Select new image", action=gtk.FILE_CHOOSER_ACTION_OPEN,
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

        f = gtk.FileFilter()
        f.set_name("Valid image files")
        f.add_mime_type("image/tiff")
        f.add_pattern("*.tiff")
        f.add_pattern("*.TIFF")
        newimg.add_filter(f)

        result = newimg.run()

        if result == gtk.RESPONSE_APPLY:
            filename= newimg.get_filename()
            self.analysis_img.set_text(filename)
            self.f_settings.image_path = newimg.get_filename()
            newimg.destroy()
            self.f_settings.marker_analysis(output_function = self.analysis_img.set_text)
            self.f_settings.set_areas_positions()

            self.owner.DMS("Reference scan positions", 
                str(self.f_settings.fixture_config_file.get("grayscale_area")), level = 110)
            self.owner.DMS("Scan positions", 
                str(self.f_settings.current_analysis_image_config.get("grayscale_area")), level = 110)

            if self.f_settings.A != None:
                dpi_factor = 4.0
                self.f_settings.A.load_other_size(filename, dpi_factor)
                grayscale = self.f_settings.A.get_subsection(self.f_settings.current_analysis_image_config.get("grayscale_area"))

                #EMPTYING self.plots_vbox
                for child in self.plots_vbox.get_children():
                    self.plots_vbox.remove(child)

                if grayscale != None:
                    #GRAYSCALE PLOT 

                    gs = img_base.Analyse_Grayscale(image=grayscale)

                    label = gtk.Label("Grayscale analysis")
                    label.show()
                    self.plots_vbox.pack_start(label, False, False, 2)
                    grayscale_fig = plt.Figure(figsize=(50,40), dpi=100)
                    grayscale_fig.subplots_adjust(left=0.02, right=0.98, wspace=0.3)
                    grayscale_plot = grayscale_fig.add_subplot(121)
                    grayscale_plot.imshow(grayscale.T)
                    Y = np.ones(len(gs._grayscale_pos)) * gs._mid_orth_strip 
                    grayscale_plot.plot(gs._grayscale_pos, Y,'ko', mfc='w', mew=1, ms=3)
                    grayscale_plot.get_xaxis().set_visible(False)
                    grayscale_plot.get_yaxis().set_visible(False)
                    grayscale_plot.set_xlim(xmin=0,xmax=grayscale.shape[0])

                    grayscale_plot = grayscale_fig.add_subplot(122)

                    z2 = np.polyfit(gs._grayscale_X, gs._grayscale,2)
                    p2 = np.poly1d(z2)
                    z3 = np.polyfit(gs._grayscale_X, gs._grayscale,3)
                    p3 = np.poly1d(z3)

                    xp = np.linspace(gs._grayscale_X[0], gs._grayscale_X[-1], 100)
                    line1 = grayscale_plot.plot(gs._grayscale_X, gs._grayscale,'b.', mfc='None', mew=2)
                    #line2 = grayscale_plot.plot(xp, p2(xp),'r-')
                    line3 = grayscale_plot.plot(xp, p3(xp),'g-')

                    #grayscale_plot.legend( (line1, line2, line3),
                       #('Data', 'Poly-2', 'Poly-3'),
                       #'upper right' )


                    grayscale_plot.axis("tight")
                    grayscale_plot.get_xaxis().set_visible(False)

                    grayscale_canvas = FigureCanvas(grayscale_fig)
                    grayscale_canvas.show()
                    grayscale_canvas.set_size_request(400,150)
                    self.plots_vbox.pack_start(grayscale_canvas, False, False, 2)


                #ENTIRE SCAN PLOT
                label = gtk.Label("Marker detection analysis")
                label.show()
                self.plots_vbox.pack_start(label, False, False, 2)
                figsize = (500,350)
                self.image_fig = plt.Figure(figsize=figsize, dpi=150)
                image_plot = self.image_fig.add_subplot(111)
                image_canvas = FigureCanvas(self.image_fig)
                self.image_fig.canvas.mpl_connect('button_press_event', self.plot_click)
                self.image_fig.canvas.mpl_connect('button_release_event', self.plot_release)
                self.image_fig.canvas.mpl_connect('motion_notify_event', self.plot_move)

                image_plot.imshow(self.f_settings.A._img.T)
                ax = image_plot.plot(self.f_settings.mark_Y*dpi_factor, self.f_settings.mark_X*dpi_factor,'ko', mfc='None', mew=2)
                self.selection_rect = plt_patches.Rectangle(
                        (0,0),0,0, ec = 'k', fill=False, lw=0.2
                        )
                self.selection_rect.get_axes()
                self.selection_rect.get_transform()
                image_plot.add_patch(self.selection_rect)
                image_plot.get_xaxis().set_visible(False)
                image_plot.get_yaxis().set_visible(False)
                image_plot.set_xlim(xmin=0,xmax=self.f_settings.A._img.shape[0])
                image_plot.set_ylim(ymin=0,ymax=self.f_settings.A._img.shape[1])
                image_canvas.show()
                image_canvas.set_size_request(figsize[0],figsize[1])

                self.plots_vbox.pack_start(image_canvas, False, False, 2)




            self.analysis_img.set_text('Ready to use: ' + str(filename))
            self.section_picking.set_text("Click on upper left corner of interest.")

    def plot_click(self, event=None):

        self._rect_ul = (event.xdata, event.ydata)
        self.selection_rect.set_x(event.xdata)
        self.selection_rect.set_y(event.ydata)
        self._rect_marking = True

    def plot_move(self, event=None):

        if self._rect_marking and event.xdata != None and event.ydata != None:
            #self.selection_rect = plt_patches.Rectangle(
                #self.plot_ul , 
            self.selection_rect.set_width(    event.xdata - self._rect_ul[0])#,
            self.selection_rect.set_height(    event.ydata - self._rect_ul[1])#,
                #ec = 'k', fc='b', fill=True, lw=1,
                #axes = self.image_ax)
            self.image_fig.canvas.draw() 
            self.owner.DMS("SELECTING", "Selecting something in the image", 1)

    def plot_release(self, event=None):

        if self._rect_marking:
            if event.xdata and event.ydata and self._rect_ul[0] and self._rect_ul[1]:
                self._rect_lr = (event.xdata, event.ydata)
                self.owner.DMS("SELECTION", "UL: " + str(self._rect_ul) + ", LR: (" + 
                    str(event.xdata) + ", "  +
                    str(event.ydata) + ")", level=1)

                self.get_analysis()

        self._rect_marking = False

    def get_analysis(self):

        #EMPTYING self.plots_vbox
        for child in self.plots_vbox2.get_children():
            self.plots_vbox2.remove(child)

        #Getting things in order        
        if self._rect_ul[0] < self._rect_lr[0]:
            upper = self._rect_ul[0]
            lower = self._rect_lr[0]
        else:
            lower = self._rect_ul[0]
            upper = self._rect_lr[0]
        if self._rect_ul[1] < self._rect_lr[1]:
            left = self._rect_ul[1]
            right = self._rect_lr[1]
        else:
            right = self._rect_ul[1]
            left = self._rect_lr[1]

        img_section = self.f_settings.A._img[upper:lower,left:right]
        x_factor = img_section.shape[0] / 200
        y_factor = img_section.shape[1] / 300

        if x_factor > y_factor:
            scale_factor = x_factor
        else:
            scale_factor = y_factor
        if scale_factor == 0:
            scale_factor = 1

        image_size = (img_section.shape[0]/scale_factor,
            img_section.shape[1]/scale_factor)

        label = gtk.Label("Selection:")
        label.show()
        self.plots_vbox2.pack_start(label, False, False, 2)
 
        image_fig = plt.Figure(figsize=image_size, dpi=150)
        image_plot = image_fig.add_subplot(111)
        image_canvas = FigureCanvas(image_fig)
        image_ax = image_plot.imshow(img_section)
        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)
        image_plot.set_xlim(xmin=0,xmax=img_section.shape[1])
        image_plot.set_ylim(ymin=0,ymax=img_section.shape[0])
        image_canvas.show()
        image_canvas.set_size_request(image_size[1], image_size[0])
        self.plots_vbox2.pack_start(image_canvas, False, False, 2)

        cell = colonies.get_grid_cell_from_array(img_section)
        features = cell.get_analysis()

        blob = cell.get_item('blob')
        blob_filter = blob.filter_array
        blob_hist = blob.histogram

        label = gtk.Label("Blob vs Background::")
        label.show()
        self.plots_vbox2.pack_start(label, False, False, 2)

        image_fig = plt.Figure(figsize=image_size, dpi=150)
        image_plot = image_fig.add_subplot(111)
        image_canvas = FigureCanvas(image_fig)
        image_ax = image_plot.imshow(blob_filter)
        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)
        image_plot.set_xlim(xmin=0,xmax=blob_filter.shape[1])
        image_plot.set_ylim(ymin=0,ymax=blob_filter.shape[0])
        image_canvas.show()
        image_canvas.set_size_request(image_size[1], image_size[0])
        self.plots_vbox2.pack_start(image_canvas, False, False, 2)

        bincenters = 0.5*(blob_hist.labels[1:]+blob_hist.labels[:-1])
        
        label = gtk.Label("Histogram:")
        label.show()
        self.plots_vbox2.pack_start(label, False, False, 2)

        image_fig = plt.Figure(figsize=(400,200), dpi=150)
        image_plot = image_fig.add_subplot(111)
        image_canvas = FigureCanvas(image_fig)
        image_plot.bar(blob_hist.labels, blob_hist.counts)
        image_plot.axvline(blob.threshold, c='r')
        
        if features != None:
            self.cell_area.set_text(str(features['cell']['area']))
            
            self.bg_mean.set_text(str(features['background']['mean']))
            self.bg_iqr_mean.set_text(str(features['background']['IQR_mean']))
            self.bg_median.set_text(str(features['background']['median']))
            image_plot.axvline(features['background']['mean'], c='g')

            self.colony_size.set_text(str(abs(features['cell']['pixelsum'] - \
                features['background']['mean'] * features['cell']['area'])))
            self.owner.DMS('Analysis', 'Alternative measures: ' + 
                "see in program"+ " (from mean), " + 
                str(abs(features['cell']['pixelsum'] - \
                features['background']['median'] * features['cell']['area'])) + 
                " (from median)", level = 111)


        image_canvas.set_size_request(400, 300)
        image_canvas.show()
        self.plots_vbox2.pack_start(image_canvas, False, False, 2)

        label = gtk.Label("Threshold (red), Background Mean(green)")
        label.show()
        self.plots_vbox2.pack_start(label, False, False, 2)

