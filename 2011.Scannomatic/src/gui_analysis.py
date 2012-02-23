#!/usr/bin/env python
"""The GTK-GUI view and its functions"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL"
__version__ = "3.0"
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
import matplotlib.patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

#
# SCANNOMATIC LIBRARIES
#

import src.resource_image as img_base
import src.resource_fixture as fixture_settings
import src.analysis_wrapper as colonies

#
# CLASSES
#

class Analyse_One(gtk.Frame):

    def __init__(self, owner, label="Analyse One Image"):

        gtk.Frame.__init__(self, label)

        self.owner = owner
        self.analysis = None

        self._rect_marking = False
        self._rect_ul = None
        self._rect_lr = None
        self._circ_marking = False
        self._circ_dragging = False
        self._circ_center = None

        self._config_calibration_path = self.owner._program_config_root + os.sep + "calibration.data"
        self._config_calibration_polynomial = self.owner._program_config_root + os.sep + "calibration.polynomials"

        self._fixture_config_root = self.owner._program_config_root + os.sep + "fixtures"
        self.f_settings = fixture_settings.Fixture_Settings(self._fixture_config_root, fixture="fixture_a")

        vbox = gtk.VBox()
        vbox.show()
        self.add(vbox)

        hbox = gtk.HBox()
        hbox.show()
        vbox.pack_start(hbox, False, False, 2)

        #
        # Main image and gray scale
        #

        self.plots_vbox = gtk.VBox()
        self.plots_vbox.show()
        hbox.pack_start(self.plots_vbox, False, False, 2)
        

        label = gtk.Label("Grayscale analysis")
        label.show()
        self.plots_vbox.pack_start(label, False, False, 2)

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
        self.plots_vbox.pack_start(grayscale_canvas, False, False, 2)


        label = gtk.Label("Marker detection analysis")
        label.show()
        self.plots_vbox.pack_start(label, False, False, 2)

        figsize = (500,350)

        self.image_fig = plt.Figure(figsize=figsize, dpi=75)
        image_plot = self.image_fig.add_subplot(111)
        image_canvas = FigureCanvas(self.image_fig)
        self.image_fig.canvas.mpl_connect('button_press_event', self.plot_click)
        self.image_fig.canvas.mpl_connect('button_release_event', self.plot_release)
        self.image_fig.canvas.mpl_connect('motion_notify_event', self.plot_drag)

        self.selection_rect = plt_patches.Rectangle(
                (0,0),0,0, ec = 'k', fill=False, lw=0.5
                )
        #self.selection_rect.get_axes()
        #self.selection_rect.get_transform()

        image_plot.add_patch(self.selection_rect)
        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)
        image_canvas.show()
        image_canvas.set_size_request(figsize[0],figsize[1])

        self.plots_vbox.pack_start(image_canvas, False, False, 2)


        self.gs_reset_button = gtk.Button(label = 'No image loaded...')
        self.gs_reset_button.show()
        self.gs_reset_button.connect("clicked", self.set_grayscale_selecting)
        self.plots_vbox.pack_start(self.gs_reset_button, False, False, 2)
        self.gs_reset_button.set_sensitive(False)
        #
        # Plot sections / blob images
        #

        self.selection_circ = plt_patches.Circle((0,0),0)
        self._blobs_have_been_loaded = False

        #
        # Other
        #

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

        label = gtk.Label("Manual selection size:")
        label.show()
        vbox2.pack_start(label, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox2.pack_start(hbox, False, False, 2)

        self.selection_width = gtk.Entry()
        self.selection_width.show()
        self.selection_width.set_text("")
        self.selection_width.connect("focus-out-event", self.manual_selection_width)
        hbox.pack_start(self.selection_width, False, False, 2)
      
        label = gtk.Label("x")
        label.show()
        hbox.pack_start(label, False, False, 2)
 
        self.selection_height = gtk.Entry()
        self.selection_height.show()
        self.selection_height.set_text("")
        self.selection_height.connect("focus-out-event", self.manual_selection_height)
        hbox.pack_start(self.selection_height, False, False, 2)

        #Analysis data frame for selection
        frame = gtk.Frame("Image")
        frame.show()
        vbox2.pack_start(frame, False, False, 2)

        vbox3 = gtk.VBox()
        vbox3.show()
        frame.add(vbox3)

        #Interactive helper
        self.section_picking = gtk.Label("First load an image.")
        self.section_picking.show()
        vbox3.pack_start(self.section_picking, False, False, 10)

        #
        # KODAK VALUE SPACE
        #

        #Analysis data frame for selection
        frame = gtk.Frame("'Kodak Value Space'")
        frame.show()
        vbox2.pack_start(frame, False, False, 2)

        vbox3 = gtk.VBox()
        vbox3.show()
        frame.add(vbox3)

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

        #Blob area
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Area:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.blob_area = gtk.Label("0")
        self.blob_area.show()
        hbox.pack_end(self.blob_area, False, False, 2)

        #Blob Size
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Size:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.colony_size = gtk.Label("0")
        self.colony_size.show()
        hbox.pack_end(self.colony_size, False, False, 2)

        #Blob Pixelsum 
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Pixelsum:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.blob_pixelsum = gtk.Label("0")
        self.blob_pixelsum.show()
        hbox.pack_end(self.blob_pixelsum, False, False, 2)

        #Blob Mean 
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Mean:")
        label.show()
        hbox.pack_start(label,False, False, 2)

        self.blob_mean = gtk.Label("0")
        self.blob_mean.show()
        hbox.pack_end(self.blob_mean, False, False, 2)

        #
        # CELL ESTIMATE SPACE
        #

        #Cell Count Estimations
        frame = gtk.Frame("Cell Estimate Space")
        frame.show()
        vbox2.pack_start(frame, False, False, 2)

        vbox3 = gtk.VBox()
        vbox3.show()
        frame.add(vbox3)

        #Unit
        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        self.cce_per_pixel = gtk.Label("0")
        self.cce_per_pixel.show()
        hbox.pack_start(self.cce_per_pixel)

        label = gtk.Label("depth/pixel")
        label.show()
        hbox.pack_end(label, False, False, 2)

        label = gtk.Label("Independent measure:")
        label.show()
        vbox3.pack_start(label, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("CCE/grid-cell:")
        label.show()
        hbox.pack_start(label, False, False, 2)

        self.cce_indep_measure = gtk.Entry()
        self.cce_indep_measure.connect("focus-out-event", self.verify_number)
        self.cce_indep_measure.show()
        hbox.pack_end(self.cce_indep_measure, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label('Data point label:')
        label.show()
        hbox.pack_start(label, False, False, 2)

        self.cce_data_label = gtk.Entry()
        self.cce_data_label.show()
        hbox.pack_end(self.cce_data_label, False, False, 2)      

        button = gtk.Button("Submit calibration point")
        button.show()
        button.connect("clicked", self.add_calibration_point, None)
        vbox3.pack_start(button, False, False, 2)

        self.cce_calculated = gtk.Label("--- cells in blob")
        self.cce_calculated.show()
        vbox3.pack_start(self.cce_calculated, False, False, 2)
        self._cce_poly_coeffs = None
        has_poly_cal = True
        try:
            fs = open(self._config_calibration_polynomial, 'r')
        except:
            has_poly_cal = False
        if has_poly_cal:
            self._cce_poly_coeffs = []
            for l in fs:
                l_data = eval(l.strip("\n"))
                if type(l_data) == types.ListType:
                    self._cce_poly_coeffs = l_data[-1]
                    break
            label = gtk.Label("(using '" + str(l_data[0]) + "')")
            label.show()
            vbox3.pack_start(label, False, False, 2)
            fs.close()

        self.blob_filter = None

    def verify_number(self, widget=None, event=None, data=None):

        try:
            float(widget.get_text())
        except:
            widget.set_text("0")

    def get_vector_polynomial_sum_single(self, X, coefficient_array):

        return np.sum(np.polyval(coefficient_array, X))


    def get_expanded_vector(self, compressed_vector):

        vector = []

        for pos in xrange(len(compressed_vector[0])):

            for ith in xrange(compressed_vector[1][pos]):

                vector.append(compressed_vector[0][pos])

        return vector

    def get_cce_data_vector(self):

        if self.blob_filter != None and self._rect_ul != None and self._rect_lr != None:

            img_section = self.get_img_section(self._rect_ul, self._rect_lr, as_copy=True)


            tf_matrix = np.asarray(colonies.get_gray_scale_transformation_matrix(self._grayscale))

            if tf_matrix is not None:
                for x in xrange(img_section.shape[0]):
                    for y in xrange(img_section.shape[1]):
                        img_section[x,y] = tf_matrix[img_section[x,y]]


            ###DEBUG CALIBRATION VALUES
            #print "REF: area", self.blob_image[np.where(self.blob_filter)].size
            #print "REF pixsum", self.blob_image[np.where(self.blob_filter)].sum()
            #print "--"
            #print "img_section == orig_tf_img (0==True)", (img_section - self.img_transf).sum() 
            #print "--"
            #print "CAL: area", img_section[np.where(self.blob_filter)].size
            #print "CAL: pixsum", img_section[np.where(self.blob_filter)].sum()
            ###DEBUG END

            #Get the effect of blob-materia on pixels
            blob_pixels = img_section[np.where(self.blob_filter)] - float(self.bg_mean.get_text())


            #Disallow "anti-materia" pixels           
            blob_pixels = blob_pixels[np.where(blob_pixels > 0)] 


            keys = list(np.unique(blob_pixels))
            values = []
            for k in keys:
                values.append(np.sum(blob_pixels == k))

            cce_data = [keys, values]

            return cce_data

    def add_calibration_point(self, widget=None, event=None, data=None):

        if self.blob_filter != None and self._rect_ul != None and self._rect_lr != None:

            indep_cce = float(self.cce_indep_measure.get_text()) 

            cce_data = self.get_cce_data_vector()

            try:
                fs = open(self._config_calibration_path,'a')



                fs.write(str([self.analysis_img.get_text(), self.cce_data_label.get_text() ,indep_cce, cce_data]) + "\n")

                fs.close()
                
            except:

                self.owner.DMS("Error", "Could not open " + self._config_calibration_path)
                return

            self.owner.DMS("Calibration", "Setting " + self.analysis_img.get_text() + \
                " colony depth per pixel to " + str( indep_cce ) + " cell estimate.")

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
            self._grayscale = None
            filename= newimg.get_filename()
            self.analysis_img.set_text(filename)
            self.f_settings.image_path = newimg.get_filename()
            newimg.destroy()
            self.f_settings.marker_analysis(output_function = self.analysis_img.set_text)



            fixture_X, fixture_Y = self.f_settings.get_fixture_markings()

            if len(fixture_X) == len(self.f_settings.mark_X):
                self.f_settings.set_areas_positions()

                self.owner.DMS("Reference scan positions", 
                    str(self.f_settings.fixture_config_file.get("grayscale_area")), level = 110)
                self.owner.DMS("Scan positions", 
                    str(self.f_settings.current_analysis_image_config.get("grayscale_area")), level = 110)

                rotated = True
            else:
                self.owner.DMS("Error", "Missmatch between fixture configuration and current image", level =110)
                rotated = False

            if self.f_settings.A != None:


                grayscale = None
                dpi_factor = 4.0
                self.f_settings.A.load_other_size(filename, dpi_factor)
                if rotated:
                    grayscale = self.f_settings.A.get_subsection(self.f_settings.current_analysis_image_config.get("grayscale_area"))
                #EMPTYING self.plots_vbox
                #for child in self.plots_vbox.get_children():
                #    self.plots_vbox.remove(child)

                if grayscale != None:

                    gs_success = self.set_grayscale(grayscale)

                if grayscale is None or gs_success == False:
                    self.set_grayscale_selecting()

                #THE LARGE IMAGE

                image_plot = self.image_fig.gca()
                image_plot.cla()
                image_plot.imshow(self.f_settings.A._img.T)
                ax = image_plot.plot(self.f_settings.mark_Y*dpi_factor, self.f_settings.mark_X*dpi_factor,'ko', mfc='None', mew=2)
                image_plot.set_xlim(xmin=0,xmax=self.f_settings.A._img.shape[0])
                image_plot.set_ylim(ymin=0,ymax=self.f_settings.A._img.shape[1])
                image_plot.add_patch(self.selection_rect)
                self.image_fig.canvas.draw()

            self.analysis_img.set_text(str(filename))
            self.section_picking.set_text("Select area (and guide blob-detection if needed).")

            if self._blobs_have_been_loaded:
                self.blob_fig.cla()
                self.blob_bool_fig.cla()
                self.blob_hist.cla()


            if self.selection_rect.get_width() > 0 and self.selection_rect.get_height() > 0:

                self.get_analysis()

    def set_grayscale_selecting(self, widget=None, event=None, data=None):

        self.gs_reset_button.set_sensitive(False)
        self.gs_reset_button.set_label("Currently selecting a gray-scale area")
        self._grayscale = None
        self.grayscale_plot_img.clear()
        self.grayscale_plot.clear()
        self.grayscale_fig.canvas.draw()


    def set_manual_grayscale(self, ul, lr):

        img_section = self.get_img_section(ul, lr, as_copy=False)
        self.set_grayscale(img_section)

    def set_grayscale(self, im_section):

        self.gs_reset_button.set_sensitive(True)
        self.gs_reset_button.set_label("Click to reset grayscale")

        gs = img_base.Analyse_Grayscale(image=im_section)
        self._grayscale = gs._grayscale

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
            self.owner.DMS("Error", "There's something wrong with the grayscale. Switching to manual")
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

    def get_click_in_rect(self, event):

        if self._rect_lr is None or self._rect_ul is None:
            return False

        low_x = np.array([self._rect_lr[0], self._rect_ul[0]]).min()
        high_x = np.array([self._rect_lr[0], self._rect_ul[0]]).max()
        low_y = np.array([self._rect_lr[1], self._rect_ul[1]]).min()
        high_y = np.array([self._rect_lr[1], self._rect_ul[1]]).max()

        if low_x < event.xdata < high_x and low_y < event.ydata < high_y:

            return True
 
        return False

    def get_click_in_circle(self, event):

        r = self.selection_circ.get_radius()
        cur_pos = np.asarray((event.xdata, event.ydata))
       
        if self._circ_center is None:
            return False
 
        if r**2 >= np.sum ( (cur_pos - self._circ_center)**2 ):

            return True

        else:

            return False


    def manual_selection_width(self, widget=None, event=None, data=None):
        try:
            self.selection_rect.set_width(float(widget.get_text()))
        except:
            widget.set_text("")

        self.image_fig.canvas.draw()
        self.get_analysis()

    def manual_selection_height(self, widget=None, event=None, data=None):
        try:
            self.selection_rect.set_height(float(widget.get_text()))
        except:
            widget.set_text("")

        self.image_fig.canvas.draw()
        self.get_analysis()

    def plot_click(self, event=None):

        if self._rect_marking == True:
            self.plot_release(event)
        else:
            if self.get_click_in_rect(event) == False:
                self._rect_dragging = False
                self._rect_ul = (event.xdata, event.ydata)
                self.selection_rect.set_x(event.xdata)
                self.selection_rect.set_y(event.ydata)
                self._rect_marking = True
            else:
                self._rect_marking = True
                self._rect_dragging = True
                self._dragging_origin = np.asarray((event.xdata, event.ydata))
                self._dragging_rect_origin = np.asarray((\
                    self.selection_rect.get_x(), 
                    self.selection_rect.get_y()))

    def blob_click(self, event=None):

        if self._circ_marking == True:
            self.plot_release(event)
        else:
            if self.get_click_in_circle(event) == False:

                self._circ_center = np.asarray((event.xdata, event.ydata))
                self.selection_circ.center = (event.xdata, event.ydata)

                self._circ_dragging = False
                self._circ_marking = True

            else:

                self._circ_marking = True
                self._circ_dragging = True

                self._dragging_origin = np.asarray((event.xdata, event.ydata))
                self._dragging_circ_origin = np.asarray(\
                    self.selection_circ.center)

    def plot_drag(self, event=None):

        if self._rect_marking and event.xdata != None and event.ydata != None:
            if self._rect_dragging == False:
                #self.selection_rect = plt_patches.Rectangle(
                    #self.plot_ul , 
                self.selection_rect.set_width(    event.xdata - self._rect_ul[0])#,
                self.selection_rect.set_height(    event.ydata - self._rect_ul[1])#,
                    #ec = 'k', fc='b', fill=True, lw=1,
                    #axes = self.image_ax)
                self.owner.DMS("SELECTING", "Selecting something in the image", 1)

            else:
                cur_pos_offset = np.asarray((event.xdata, event.ydata)) - \
                    self._dragging_origin
                new_rect_pos = self._dragging_rect_origin + cur_pos_offset
                self.selection_rect.set_x(new_rect_pos[0])
                self.selection_rect.set_y(new_rect_pos[1])
                self.owner.DMS("SELECTING", "Moving selection", 1)

            self.image_fig.canvas.draw() 

    def blob_drag(self, event=None):

        if self._circ_marking and event.xdata != None and event.ydata != None:

            cur_pos = np.asarray((event.xdata, event.ydata))

            if self._circ_dragging == False:

                r = np.sqrt( np.sum( (cur_pos - self._circ_center)**2 ) )

                self.selection_circ.set_radius( r )

                self.owner.DMS("SELECTING", "Selecting some blob", 1)

            else:
                cur_pos_offset = cur_pos - \
                    self._dragging_origin
                new_circ_pos = self._dragging_circ_origin + cur_pos_offset
                self.selection_circ.center = tuple(new_circ_pos)
                self.owner.DMS("SELECTING", "Moving blob selection", 1)

            self.blob_fig.canvas.draw() 

    def plot_release(self, event=None):

        if self._rect_marking:
            self._rect_marking = False
            if self._rect_dragging == False:
                if event.xdata and event.ydata and self._rect_ul[0] and self._rect_ul[1]:
                    self._rect_lr = (event.xdata, event.ydata)
                    self.owner.DMS("SELECTION", "UL: " + str(self._rect_ul) + ", LR: (" + 
                        str(event.xdata) + ", "  +
                        str(event.ydata) + ")", level=1)

                    self.selection_width.set_text(str(self.selection_rect.get_width()))
                    self.selection_height.set_text(str(self.selection_rect.get_height()))
                    if self._grayscale != None:
                        self.get_analysis()
                    else:
                        self.set_manual_grayscale(self._rect_ul, self._rect_lr)

            else:
                self._rect_ul = (self.selection_rect.get_x(),
                    self.selection_rect.get_y())
                self._rect_lr = (self._rect_ul[0] + self.selection_rect.get_width(),
                    self._rect_ul[1] + self.selection_rect.get_height())

                self.owner.DMS("SELECTION", "UL: " + str(self._rect_ul) + ", LR: (" + 
                    str(self._rect_lr) + ")", level=1)

                self.get_analysis()
                

    def blob_release(self, event=None):
        if self._circ_marking:
  
            self._circ_marking = False

            if event.xdata is not None and event.ydata is not None:

                cur_pos = np.asarray((event.xdata, event.ydata))

                if self._circ_dragging == False:
                    if event.xdata and event.ydata:

                        r = np.sqrt( np.sum((cur_pos - self._circ_center)**2) )

                        self.owner.DMS("SELECTION", "Radius: " + str(r) + ")", level=1)

                        print "C center", self.selection_circ.center, "radius", self.selection_circ.get_radius()

                        self.get_analysis(center=self._circ_center, radius=r)
                else:

                    self._circ_center = self._circ_center + \
                        (cur_pos - self._dragging_origin)

                    self.selection_circ.center = tuple(self._circ_center)


                    self.get_analysis(center=self._circ_center, \
                        radius = self.selection_circ.get_radius())
                

    def get_img_section(self, ul, lr, as_copy=False):


        if ul[0] < lr[0]:
            upper = ul[0]
            lower = lr[0]
        else:
            lower = ul[0]
            upper = lr[0]

        if ul[1] < lr[1]:
            left = ul[1]
            right = lr[1]
        else:
            right = ul[1]
            left = lr[1]

        ###DEBUG SELECTION SHAPE
        print "Selection shape: ", self.f_settings.A._img[upper:lower,left:right].shape
        ###DEBUG END

        if as_copy:
            return np.copy(self.f_settings.A._img[upper:lower,left:right])
        else:
            return self.f_settings.A._img[upper:lower,left:right]

    def get_analysis(self, center=None, radius=None):

        if radius is None:
            self.selection_circ.set_radius(0)
            
        #EMPTYING self.plots_vbox
        #for child in self.plots_vbox2.get_children():
        #    self.plots_vbox2.remove(child)


        img_section = self.get_img_section(self._rect_ul, self._rect_lr, as_copy=True)
        img_transf = img_section.copy()

        tf_matrix = colonies.get_gray_scale_transformation_matrix(self._grayscale)

        if tf_matrix is not None:
            for x in xrange(img_transf.shape[0]):
                for y in xrange(img_transf.shape[1]):
                    img_transf[x,y] = tf_matrix[img_transf[x,y]]

        ###DEBUG TRANSFORMATIONS
        #print "*** SECTION TRANSFORMED: ", not(img_transf.sum() == img_section.sum())
        #self.img_transf = img_transf.copy()
        ###END DEBUG

        x_factor = img_section.shape[0] / 200
        y_factor = img_section.shape[1] / 200

        if x_factor > y_factor:
            scale_factor = x_factor
        else:
            scale_factor = y_factor
        if scale_factor == 0:
            scale_factor = 1



        #
        # BLOB SELECTION CANVAS
        #
        image_size = (img_section.shape[1]/scale_factor,
            img_section.shape[0]/scale_factor)

        if self._blobs_have_been_loaded == False:


            hbox = gtk.HBox()
            hbox.show()
            self.plots_vbox2.pack_start(hbox, False, False, 0)

            vbox = gtk.VBox()
            vbox.show()

            label = gtk.Label("Selection:")
            label.show()
            vbox.pack_start(label, False, False, 2)

            self.blob_fig = plt.Figure(figsize=image_size, dpi=150)
            self.blob_fig.add_axes()
            image_plot = self.blob_fig.gca()
            #image_plot = self.blob_fig.add_subplot(111)
            
            image_canvas = FigureCanvas(self.blob_fig)
            self.blob_fig.canvas.mpl_connect('button_press_event', self.blob_click )
            self.blob_fig.canvas.mpl_connect('button_release_event', self.blob_release)
            self.blob_fig.canvas.mpl_connect('motion_notify_event', self.blob_drag)

            image_plot.get_xaxis().set_visible(False)
            image_plot.get_yaxis().set_visible(False)

            image_canvas.show()
            image_canvas.set_size_request(image_size[1], image_size[0])
            vbox.pack_start(image_canvas, False, False, 2)
            hbox.pack_start(vbox, False, False, 2)

        if center is None and radius is None:
            image_plot = self.blob_fig.gca()
            image_plot.cla()
            image_ax = image_plot.imshow(img_section.T)
            image_plot.set_xlim(xmin=0,xmax=img_section.shape[0])
            image_plot.set_ylim(ymin=0,ymax=img_section.shape[1])

        #print "C center", self.selection_circ.center, "radius", self.selection_circ.get_radius()
            image_plot.add_patch(self.selection_circ)
        #if center is not None and radius is not None:        
        #    self.selection_circ.center = center
        #    self.selection_circ.set_radius(radius)

        self.blob_fig.canvas.draw()
 

        #
        # RETRIEVING ANALYSIS
        #

        cell = colonies.get_grid_cell_from_array(img_transf, center=center, radius = radius)

        features = cell.get_analysis(no_detect=True)

        blob = cell.get_item('blob')
        self.blob_filter = blob.filter_array
        self.blob_image = blob.grid_array

        #
        # CALCULATING CCE IF POLY-COEFFS EXISTS
        #

        if self._cce_poly_coeffs is not None:

            cce_vector = self.get_cce_data_vector()
            cce_vector = self.get_expanded_vector(cce_vector)
            cce_calculated = self.get_vector_polynomial_sum_single(cce_vector, self._cce_poly_coeffs)
            self.cce_calculated.set_text(str(cce_calculated) + " cells in blob")
        ###DEBUG IMAGE STAYS TRUE
        #print "DIFF:",  (self.blob_image - self.img_transf).sum()
        ###DEBUG END

        blob_hist = blob.histogram

        #
        # BLOB vs BACKGROUND CANVAS
        #

        if self._blobs_have_been_loaded == False:

            vbox = gtk.VBox()
            vbox.show()

            label = gtk.Label("Blob vs Background::")
            label.show()
            vbox.pack_start(label, False, False, 2)

            self.blob_bool_fig = plt.Figure(figsize=image_size, dpi=150)
            #image_plot = self.blob_bool_fig.add_subplot(111)
            image_canvas = FigureCanvas(self.blob_bool_fig)
            image_canvas.show()
            image_canvas.set_size_request(image_size[1], image_size[0])

            vbox.pack_start(image_canvas, False, False, 2)
            hbox.pack_start(vbox, False, False, 2)

            self.blob_bool_fig.add_axes()
            image_plot = self.blob_bool_fig.gca()
            image_plot.get_xaxis().set_visible(False)
            image_plot.get_yaxis().set_visible(False)
        else:
            image_plot = self.blob_bool_fig.gca()
            image_plot.cla()

        #image_ax = image_plot.imshow(img_section.T)
        blob_filter_view = self.blob_filter.astype(float) * 256 + img_transf
        blob_filter_view = blob_filter_view * ( 256/ float(np.max(blob_filter_view)) )
        image_ax = image_plot.imshow(blob_filter_view.T)

        image_plot.set_xlim(xmin=0,xmax=self.blob_filter.shape[0])
        image_plot.set_ylim(ymin=0,ymax=self.blob_filter.shape[1])


        self.blob_bool_fig.canvas.draw()

        if blob_hist.labels != None:
            bincenters = 0.5*(blob_hist.labels[1:]+blob_hist.labels[:-1])

            if self._blobs_have_been_loaded == False:
            
                label = gtk.Label("Histogram:")
                label.show()
                self.plots_vbox2.pack_start(label, False, False, 2)

                self.blob_hist = plt.Figure(figsize=image_size, dpi=150)
                self.blob_hist.add_axes()
                
                image_canvas = FigureCanvas(self.blob_hist)
                image_plot = self.blob_hist.gca()

                image_canvas.set_size_request(image_size[1], image_size[0])
                image_canvas.show()
                self.plots_vbox2.pack_start(image_canvas, False, False, 2)

                #self.blob_hist.subplots_adjust(top=2, bottom=2)

                label = gtk.Label("Threshold (red), Background Mean(green)")
                label.show()
                self.plots_vbox2.pack_start(label, False, False, 2)
            else:
                image_plot = self.blob_hist.gca()
                image_plot.cla()

            image_plot.bar(blob_hist.labels, blob_hist.counts)

            x_ticks = range(0,256,20)           
            image_plot.set_xticks(x_ticks)
            image_plot.set_xticklabels(map(str,x_ticks), fontsize='xx-small')
            image_plot.axvline(blob.threshold, c='r')
            image_plot.set_xlim(xmin=0, xmax=100)
        self._blobs_have_been_loaded = True

        if features != None:
            self.cell_area.set_text(str(features['cell']['area']))
            
            self.bg_mean.set_text(str(features['background']['mean']))
            self.bg_iqr_mean.set_text(str(features['background']['IQR_mean']))
            self.bg_median.set_text(str(features['background']['median']))

            if blob_hist.labels != None:
                image_plot.axvline(features['background']['mean'], c='g')

            self.blob_pixelsum.set_text(str(features['blob']['pixelsum']))
            self.blob_mean.set_text(str(features['blob']['mean']))
            self.blob_area.set_text(str(features['blob']['area']))

            self.colony_size.set_text(str(abs(features['cell']['pixelsum'] - \
                features['background']['mean'] * features['cell']['area'])))

            self.cce_per_pixel.set_text(str(features['blob']['mean'] - 
                features['background']['mean']))

            self.owner.DMS('Analysis', 'Alternative measures: ' + 
                "see in program"+ " (from mean), " + 
                str(abs(features['cell']['pixelsum'] - \
                features['background']['median'] * features['cell']['area'])) + 
                " (from median)", level = 111)

        self.blob_hist.canvas.draw()

