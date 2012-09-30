#!/usr/bin/env python
"""The GTK-GUI view and its functions"""
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
import src.resource_fixture_image as resource_fixture_image
import src.analysis_wrapper as colonies
import src.gui_grayscale as grayscale
#
# CLASSES
#

class Analyse_One(gtk.Frame):

    def __init__(self, owner, label="Analyse One Image"):

        gtk.Frame.__init__(self, label)

        self.owner = owner
        self.DMS = self.owner.DMS

        self.KODAK = 0
        self.CELL_ESTIMATE = 1

        self.analysis = None
        self._cell = None

        self._rect_marking = False
        self._lock_rect_dragging = False
        self._rect_ul = None
        self._rect_lr = None
        self._circ_marking = False
        self._circ_dragging = False
        self._circ_center = None

        self._config_calibration_path = self.owner._program_config_root + os.sep + "calibration.data"
        self._config_calibration_polynomial = self.owner._program_config_root + os.sep + "calibration.polynomials"

        self._fixture_config_root = self.owner._program_config_root + os.sep + "fixtures"
        self.f_settings = None
        self._current_fixture = None
        self._fixture_updating = False

        self.last_value_space = self.KODAK
        if os.path.isfile(self._config_calibration_polynomial):
            self.last_value_space = self.CELL_ESTIMATE

        #
        # GTK
        #

        main_hbox = gtk.HBox()
        self.add(main_hbox)
    
        #
        # Main image and gray scale
        #

        self.plots_vbox = gtk.VBox()
        main_hbox.pack_start(self.plots_vbox, False, False, 2)
        

        hbox = gtk.HBox()
        self.fixture = gtk.combo_box_new_text()
        self.reload_fixtures()
        self.fixture.connect("changed", self.set_fixture)
        hbox.pack_start(self.fixture, False, False, 2)
        self.plots_vbox.pack_start(hbox, False, False, 2)

        self.grayscale_frame = grayscale.Gray_Scale(self)
        self.plots_vbox.pack_start(self.grayscale_frame, False, False, 2)


        label = gtk.Label("Marker detection analysis")
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
        image_canvas.set_size_request(figsize[0],figsize[1])

        self.plots_vbox.pack_start(image_canvas, False, False, 2)


        self.gs_reset_button = gtk.Button(label = 'No image loaded...')
        self.gs_reset_button.connect("clicked", self.set_grayscale_selecting)
        self.plots_vbox.pack_start(self.gs_reset_button, False, False, 2)
        self.gs_reset_button.set_sensitive(False)


        #
        # BLOB COLUMN
        #

        self.selection_circ = plt_patches.Circle((0,0),0)
        self._blobs_have_been_loaded = False
        image_size = [100,100]
        self._no_selection = np.zeros(image_size)
    
        self.plots_vbox2 = gtk.VBox()
        main_hbox.pack_start(self.plots_vbox2, False, False, 2)

        hbox = gtk.HBox()
        self.plots_vbox2.pack_start(hbox, False, False, 0)

        #selection
        vbox = gtk.VBox()

        label = gtk.Label("Selection:")
        vbox.pack_start(label, False, False, 2)

        self.blob_fig = plt.Figure(figsize=image_size, dpi=150)
        self.blob_fig.add_axes()
        image_plot = self.blob_fig.gca()
        self.blob_fig_ax = image_plot.imshow(self._no_selection,
            cmap=plt.cm.gray_r)

        image_canvas = FigureCanvas(self.blob_fig)
        self.blob_fig.canvas.mpl_connect('button_press_event', self.blob_click )
        self.blob_fig.canvas.mpl_connect('button_release_event', self.blob_release)
        self.blob_fig.canvas.mpl_connect('motion_notify_event', self.blob_drag)

        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)

        image_canvas.set_size_request(image_size[1], image_size[0])
        vbox.pack_start(image_canvas, False, False, 2)
        hbox.pack_start(vbox, False, False, 2)

        #blob
        vbox = gtk.VBox()

        label = gtk.Label("Blob (red)::")
        vbox.pack_start(label, False, False, 2)

        self.blob_bool_fig = plt.Figure(figsize=image_size, dpi=150)

        image_canvas = FigureCanvas(self.blob_bool_fig)
        image_canvas.set_size_request(image_size[1], image_size[0])

        vbox.pack_start(image_canvas, False, False, 2)
        hbox.pack_start(vbox, False, False, 2)

        self.blob_bool_fig.add_axes()
        image_plot = self.blob_bool_fig.gca()
        self.blob_bool_fig_ax = image_plot.imshow(self._no_selection,
            vmin=0, vmax=1)
        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)

        #background
        vbox = gtk.VBox()

        label = gtk.Label("Background (red)::")
        vbox.pack_start(label, False, False, 2)

        self.bg_bool_fig = plt.Figure(figsize=image_size, dpi=150)

        image_canvas = FigureCanvas(self.bg_bool_fig)
        image_canvas.set_size_request(image_size[1], image_size[0])

        vbox.pack_start(image_canvas, False, False, 2)
        hbox.pack_start(vbox, False, False, 2)

        self.bg_bool_fig.add_axes()
        image_plot = self.bg_bool_fig.gca()
        self.bg_bool_fig_ax = image_plot.imshow(np.zeros(image_size), 
            vmin=0, vmax=1)
        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)
        
        #
        # DATA COLUMN ETC
        #

        data_vbox = gtk.VBox()
        main_hbox.pack_end(data_vbox, False, False, 2)

        hbox = gtk.HBox()
        data_vbox.pack_start(hbox, False, False, 2)

        label = gtk.Label("Select image:")
        hbox.pack_start(label, False, False, 2)

        button = gtk.Button(label = 'Open')
        button.connect("clicked", self.select_image)
        hbox.pack_end(button, False, False, 2)

        self.analysis_img = gtk.Label("")
        self.analysis_img.set_max_width_chars(40)
        self.analysis_img.set_ellipsize(pango.ELLIPSIZE_START)
        data_vbox.pack_start(self.analysis_img, False, False, 2)

        label = gtk.Label("Manual selection size:")
        data_vbox.pack_start(label, False, False, 2)

        hbox = gtk.HBox()
        data_vbox.pack_start(hbox, False, False, 2)

        self.selection_width = gtk.Entry()
        self.selection_width.set_text("")
        self.selection_width.connect("focus-out-event", self.manual_selection_width)
        hbox.pack_start(self.selection_width, False, False, 2)
      
        label = gtk.Label("x")
        hbox.pack_start(label, False, False, 2)
 
        self.selection_height = gtk.Entry()
        self.selection_height.set_text("")
        self.selection_height.connect("focus-out-event", self.manual_selection_height)
        hbox.pack_start(self.selection_height, False, False, 2)

        checkbox = gtk.CheckButton(label="Lock selection size", use_underline=False)
        checkbox.connect("clicked", self.set_lock_selection_size)
        data_vbox.pack_start(checkbox, False, False, 2)

        #Analysis data frame for selection
        frame = gtk.Frame("Image")
        data_vbox.pack_start(frame, False, False, 2)

        vbox3 = gtk.VBox()
        frame.add(vbox3)

        #Interactive helper
        self.section_picking = gtk.Label("First load an image.")
        vbox3.pack_start(self.section_picking, False, False, 10)

        frame.show_all()

        button = gtk.RadioButton(None, "Kodak Value Space")
        button2 = gtk.RadioButton(button, "Cell Estimate Space")
        button2.set_active(self.last_value_space==self.CELL_ESTIMATE)
        button.connect("toggled", self.set_value_space, self.KODAK)
        data_vbox.pack_start(button, False, False, 2)
        button2.connect("toggled", self.set_value_space, self.CELL_ESTIMATE)
        data_vbox.pack_start(button2, False, False, 2)

        
        #
        # VALUE SPACE
        #

        #Analysis data frame for selection
        self.value_space_frame = gtk.Frame("'{0} Value Space'".format(\
            ('Kodak','Cell Estimate')[self.last_value_space]))
        data_vbox.pack_start(self.value_space_frame, False, False, 2)

        vbox3 = gtk.VBox()
        self.value_space_frame.add(vbox3)

        #Cell Area
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)
     
        label = gtk.Label("Cell Area:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.cell_area = gtk.Label("0")
        self.cell_area.set_selectable(True)
        self.cell_area.set_max_width_chars(20)
        hbox.pack_end(self.cell_area, False, False, 2)

        #Background Mean
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Background Mean:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.bg_mean = gtk.Label("0")
        self.bg_mean.set_selectable(True)
        hbox.pack_end(self.bg_mean, False, False, 2)

        #Background Inter Quartile Range Mean
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Background IQR-Mean:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.bg_iqr_mean = gtk.Label("0")
        self.bg_iqr_mean.set_selectable(True)
        hbox.pack_end(self.bg_iqr_mean, False, False, 2)

        #Background Median
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Background Median:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.bg_median = gtk.Label("0")
        self.bg_median.set_selectable(True)
        hbox.pack_end(self.bg_median, False, False, 2)

        #Blob area
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Area:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.blob_area = gtk.Label("0")
        self.blob_area.set_selectable(True)
        hbox.pack_end(self.blob_area, False, False, 2)

        #Blob Size
        #hbox = gtk.HBox()
        #vbox3.pack_start(hbox, False, False, 2)

        #label = gtk.Label("Blob Size:")
        #hbox.pack_start(label,False, False, 2)

        #self.colony_size = gtk.Label("0")
        #hbox.pack_end(self.colony_size, False, False, 2)

        #Blob Pixelsum 
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Pixelsum:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.blob_pixelsum = gtk.Label("0")
        self.blob_pixelsum.set_selectable(True)
        hbox.pack_end(self.blob_pixelsum, False, False, 2)

        #Blob Mean 
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Blob Mean:")
        label.set_selectable(True)
        hbox.pack_start(label,False, False, 2)

        self.blob_mean = gtk.Label("0")
        self.blob_mean.set_selectable(True)
        hbox.pack_end(self.blob_mean, False, False, 2)
    

        #
        # CALIBRATION FRAME
        #

        #Cell Count Estimations
        self.calibration_frame = gtk.Frame("Cell Estimate Space")
        data_vbox.pack_start(self.calibration_frame, False, False, 2)

        vbox3 = gtk.VBox()
        self.calibration_frame.add(vbox3)

        #Unit
        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        self.cce_per_pixel = gtk.Label("0")
        hbox.pack_start(self.cce_per_pixel)

        label = gtk.Label("depth/pixel")
        hbox.pack_end(label, False, False, 2)

        label = gtk.Label("Independent measure:")
        vbox3.pack_start(label, False, False, 2)

        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("CCE/grid-cell:")
        hbox.pack_start(label, False, False, 2)

        self.cce_indep_measure = gtk.Entry()
        self.cce_indep_measure.connect("focus-out-event", self.verify_number)
        hbox.pack_end(self.cce_indep_measure, False, False, 2)

        hbox = gtk.HBox()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label('Data point label:')
        hbox.pack_start(label, False, False, 2)

        self.cce_data_label = gtk.Entry()
        hbox.pack_end(self.cce_data_label, False, False, 2)      

        button = gtk.Button("Submit calibration point")
        button.connect("clicked", self.add_calibration_point, None)
        vbox3.pack_start(button, False, False, 2)

        self.cce_calculated = gtk.Label("--- cells in blob")
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
            self.DMS("ANALYSIS ONE", "Using polynomial: {0}".format(\
                self._cce_poly_coeffs), "L", debug_level="info")

            vbox3.pack_start(label, False, False, 2)
            fs.close()

        self.blob_filter = None

        main_hbox.show()
        self.plots_vbox.show_all()
        data_vbox.show_all()
        self.value_space_frame.show_all()

        self.set_value_space(widget=None, value_space=self.last_value_space)        

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

    def get_kodak_image(self, lr=None, ul=None):

        if lr is None:
            lr = self._rect_lr

        if ul is None:
            ul = self._rect_ul

        if ul is not None and lr is not None:

            img_section = self.get_img_section(ul, lr, as_copy=True)

            kodak_img = np.zeros(img_section.shape, dtype=np.float64)

            self.DMS('ANALYSE ONE',
                'Selection has pixel values ({0} - {1})'.format(\
                img_section.min(), img_section.max()), level="L", debug_level="debug")

            tf_matrix = colonies.get_gray_scale_transformation_matrix(self.grayscale_frame._grayscale)

            i_min = img_section.min()
            i_max = img_section.max()

            #self.DMS('ANALYSE ONE', 
                #'Will be using the following conversions:\n{0}'.format(\
                #zip(range(i_min, i_max+1), 
                    #tf_matrix[i_min: i_max+1])),
                #110, debug_level="debug")

            if tf_matrix is not None:
                for x in xrange(img_section.shape[0]):
                    for y in xrange(img_section.shape[1]):
                        if i_max >= img_section[x,y] >= i_min:
                            kodak_img[x,y] = tf_matrix[img_section[x,y]]
                        else:
                            self.DMS('ANALYSE ONE', 'Fishy pixel at \
({0},{1}) with value {2} (using {3})'.format(x,y,img_section[x,y],
                                tf_matrix[-1]), "L", 
                                debug_level="critical")
                            kodak_img[x,y] = tf_matrix[-1] 
            else:

                return None

            self.DMS('ANALYSE ONE',
                'Selection has Kodak values ({0} - {1}), should be ({2} - {3}\
, second way {4} - {5})'\
                .format(kodak_img.min(), kodak_img.max(),
                tf_matrix[i_max], tf_matrix[i_min],
                np.array(tf_matrix[i_min: i_max+1]).min(), 
                np.array(tf_matrix[i_min: i_max+1]).max()), 
                "L", debug_level="debug")

            return kodak_img

        return None

    def get_cce_data_vector(self, img_section=None):

        if img_section is None:
            img_section = self.get_kodak_image()


        if self.blob_filter != None and img_section is not None:
            #Get the effect of blob-materia on pixels
            try:
                blob_pixels = img_section[np.where(self.blob_filter)] - float(self.bg_mean.get_text())
            except ValueError:
                self.DMS("ANALYSE ONE", "There's no background in section.",
                    "L", debug_level="warning")

                blob_pixels = img_section[np.where(self.blob_filter)]

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

                self.DMS("Error", "Could not open " + self._config_calibration_path,'LA',
                    debug_level='error')
                return

            self.DMS("Calibration", "Setting " + self.analysis_img.get_text() + \
                " colony depth per pixel to " + str( indep_cce ) + " cell estimate.", level="L",
                debug_level="info")

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
            self.grayscale_frame._grayscale = None
            filename= newimg.get_filename()
            self.analysis_img.set_text(filename)
            self.f_settings.image_path = newimg.get_filename()
            newimg.destroy()
            self.f_settings.marker_analysis(output_function = self.DMS)



            fixture_X, fixture_Y = self.f_settings.get_fixture_markings()

            if fixture_X is not None and fixture_Y is not None and \
                self.f_settings.mark_X is not None and \
                self.f_settings.mark_Y is not None:

                if len(fixture_X) == len(self.f_settings.mark_X):
                    self.f_settings.set_areas_positions()

                    self.DMS("Reference scan positions", 
                        str(self.f_settings.fixture_config_file.get("grayscale_area")), level = "L")
                    self.DMS("Scan positions", 
                        str(self.f_settings.current_analysis_image_config.get("grayscale_area")), level = "L")

                    rotated = True
                else:
                    self.DMS("Error", "Missmatch between fixture configuration and current image", 
                        level = "L", debug_level="error")
                    rotated = False

            print fixture_X, fixture_Y, self.f_settings.A

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

                    self.DMS('ANALYSE ONE', 'Automatically setting grayscale-area to {0}'.\
                        format(grayscale.shape), level="L", debug_level='debug')
                    gs_success = self.set_grayscale(grayscale)

                if grayscale is None or gs_success == False:
                    self.set_grayscale_selecting()

                #THE LARGE IMAGE

                image_plot = self.image_fig.gca()
                image_plot.cla()
                image_plot.imshow(self.f_settings.A._img.T, cmap=plt.cm.gray)
                ax = image_plot.plot(self.f_settings.mark_Y*dpi_factor, 
                    self.f_settings.mark_X*dpi_factor,'o', mfc='r', mew=2)
                image_plot.set_xlim(xmin=0,xmax=self.f_settings.A._img.shape[0])
                image_plot.set_ylim(ymin=0,ymax=self.f_settings.A._img.shape[1])
                image_plot.add_patch(self.selection_rect)
                self.image_fig.canvas.draw()

            self.analysis_img.set_text(str(filename))
            self.section_picking.set_text("Select area (and guide blob-detection if needed).")

            if self._blobs_have_been_loaded:
                self.blob_fig_ax.set_data(self._no_selection)
                self.blob_bool_fig_ax.set_data(self._no_selection)
                self.blob_hist.clf()


            if self.selection_rect.get_width() > 0 and self.selection_rect.get_height() > 0:

                self.get_analysis()

        else:

            newimg.destroy()

    def reload_fixtures(self):

        self._fixture_gui_updating = True

        directory = self._fixture_config_root
        extension = ".config"
        list_fixtures = map(lambda x: x.split(extension,1)[0], [file for file\
            in os.listdir(directory) if file.lower().endswith(extension)])


        for f in list_fixtures:

            need_input = True

            for pos in xrange(len(self.fixture.get_model())-1,-1,-1):

                cur_text = self.fixture.get_model()[pos][0]

                if cur_text == f:
                    need_input = False
                    break
                elif cur_text not in list_fixtures:
                    self.fixture.remove_text(pos)
                        
                        
            if need_input:
                self.fixture.append_text(f)

        if self._current_fixture is not None and True not in map(lambda x: x[0] ==\
            self._current_fixture, self.fixture.get_model()):


            self._current_fixture = None

        else:

            pos = next((i for i, f in enumerate(map(lambda x: x[0] == \
                self._current_fixture, self.fixture.get_model())) \
                if f is True),-1)

            self.set_fixture(data = pos)

        if self._current_fixture is None:

            if len(self.fixture.get_model()) > 0:

                self.set_fixture(data=0)

            else: 
            
                self.f_settings = fixture_settings.Fixture_Settings(self._fixture_config_root, fixture='fixture_a')
                self._current_fixture = 'fixture_a'
                self.reload_fixtures()

        self._fixture_gui_updating = False

    def set_fixture(self, widget=None, data=None):

        if not self._fixture_updating:
            self._fixture_updating = True
            if data != None:

                self.fixture.set_active(data)
                fixture = self.fixture.get_model()[data][0]
 
            else:    
                fixture = widget.get_model()[widget.get_active()][0]

            self.f_settings = fixture_settings.Fixture_Settings(\
                self._fixture_config_root, fixture=fixture)

            self._fixture_updating = False

    def set_value_space(self, widget=None, value_space=None):

        if value_space in (self.KODAK, self.CELL_ESTIMATE): 
            self.last_value_space = value_space
            self.value_space_frame.set_label("{0} Space".format(\
                ('Kodak','Cell Estimate')[self.last_value_space]))
            if self.last_value_space == self.KODAK:
                self.calibration_frame.show_all()
            else:
                self.calibration_frame.hide()


            features = self.get_features()

            if features is not None:

                self.set_features_in_gui(features)

    def set_grayscale_selecting(self, widget=None, event=None, data=None):

        self.gs_reset_button.set_sensitive(False)
        self.gs_reset_button.set_label("Currently selecting a gray-scale area")
        self.grayscale_frame._grayscale = None


    def set_manual_grayscale(self, ul, lr):

        img_section = self.get_img_section(ul, lr, as_copy=False)
        self.DMS('ANALYSE ONE', 'Manually setting grayscale-area to {0} {1} (shape {2}'.\
            format(ul, lr, img_section.shape), level="L", debug_level='debug')
        if not self.set_grayscale(img_section):
            self.set_grayscale_selecting()

    def set_grayscale(self, im_section):

        self.gs_reset_button.set_sensitive(True)
        self.gs_reset_button.set_label("Click to reset grayscale")
        self.grayscale_frame.set_grayscale(im_section)
        return self.grayscale_frame.get_has_grayscale()

    def set_lock_selection_size(self, widget = None):

        if widget.get_active():

            self._lock_rect_dragging = True
            self._rect_dragging = True        

        else:

            self._lock_rect_dragging = False

    def set_features_in_gui(self, features):

        if features != None:
            DECIMAL_TXT = "{0:.{1}f}"
            self.cell_area.set_text(str(features['cell']['area']))
           
            try:
                self.bg_mean.set_text(DECIMAL_TXT.format(\
                    features['background']['mean'], 2))
            except:
                self.bg_mean.set_text("0")

            try:
                self.bg_iqr_mean.set_text(DECIMAL_TXT.format(\
                    features['background']['IQR_mean'], 2))
            except:
                self.bg_iqr_mean.set_text("0")
            try:
                self.bg_median.set_text(DECIMAL_TXT.format(\
                    features['background']['median'], 2))
            except:
                self.bg_median.set_text("0")

            try:
                self.blob_pixelsum.set_text(DECIMAL_TXT.format(\
                    float(features['blob']['pixelsum']), 0))
            except:
                self.blob_pixelsum.set_text("0")

            try:
                self.blob_mean.set_text(DECIMAL_TXT.format(\
                    features['blob']['mean'], 2))
            except:
                self.blob_mean.set_text("0")

            self.blob_area.set_text(str(features['blob']['area']))

            #self.colony_size.set_text(str(abs(features['cell']['pixelsum'] - \
            #    features['background']['mean'] * features['cell']['area'])))

            self.cce_per_pixel.set_text(str(features['blob']['mean'] - 
                features['background']['mean']))

            self.DMS('Analysis', 'Alternative measures: ' + 
                "see in program"+ " (from mean), " + 
                str(abs(features['cell']['pixelsum'] - \
                features['background']['median'] * features['cell']['area'])) + 
                " (from median)", level = "L", debug_level='info')


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
            if self._lock_rect_dragging == False and \
                self.get_click_in_rect(event) == False:

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
                self.DMS("SELECTING", "Selecting something in the image", level="A")

            else:
                cur_pos_offset = np.asarray((event.xdata, event.ydata)) - \
                    self._dragging_origin
                new_rect_pos = self._dragging_rect_origin + cur_pos_offset
                self.selection_rect.set_x(new_rect_pos[0])
                self.selection_rect.set_y(new_rect_pos[1])
                self.DMS("SELECTING", "Moving selection", level="A")

            self.image_fig.canvas.draw() 

    def blob_drag(self, event=None):

        if self._circ_marking and event.xdata != None and event.ydata != None:

            cur_pos = np.asarray((event.xdata, event.ydata))

            if self._circ_dragging == False:

                r = np.sqrt( np.sum( (cur_pos - self._circ_center)**2 ) )

                self.selection_circ.set_radius( r )

                self.DMS("SELECTING", "Selecting some blob", "A")

            else:
                cur_pos_offset = cur_pos - \
                    self._dragging_origin
                new_circ_pos = self._dragging_circ_origin + cur_pos_offset
                self.selection_circ.center = tuple(new_circ_pos)
                self.DMS("SELECTING", "Moving blob selection", "A")

            self.blob_fig.canvas.draw() 

    def plot_release(self, event=None):

        self.DMS("ANALYSE ONE", "{0} rect marking for {1}.".format(
            ['Made','Dragged'][self._rect_dragging],
            ['gray-scale', 'feature-selection'][self.grayscale_frame._grayscale is not None]),
            level="L", debug_level = 'debug')

        if self._rect_marking:
            self._rect_marking = False
            if self._rect_dragging == False:
                if event.xdata and event.ydata and self._rect_ul[0] and self._rect_ul[1]:
                    self._rect_lr = (event.xdata, event.ydata)
                    self.DMS("SELECTION", "UL: " + str(self._rect_ul) + ", LR: (" + 
                        str(event.xdata) + ", "  +
                        str(event.ydata) + ")", level="A")

                    self.selection_width.set_text(str(self.selection_rect.get_width()))
                    self.selection_height.set_text(str(self.selection_rect.get_height()))
                    if self.grayscale_frame._grayscale != None:
                        self.get_analysis()
                    else:
                        self.set_manual_grayscale(self._rect_ul, self._rect_lr)

            else:
                self._rect_ul = (self.selection_rect.get_x(),
                    self.selection_rect.get_y())
                self._rect_lr = (self._rect_ul[0] + self.selection_rect.get_width(),
                    self._rect_ul[1] + self.selection_rect.get_height())

                self.DMS("SELECTION", "UL: " + str(self._rect_ul) + ", LR: (" + 
                    str(self._rect_lr) + ")", level="A")

                self.get_analysis()
                

    def blob_release(self, event=None):
        if self._circ_marking:
  
            self._circ_marking = False

            if event.xdata is not None and event.ydata is not None:

                cur_pos = np.asarray((event.xdata, event.ydata))

                if self._circ_dragging == False:
                    if event.xdata and event.ydata:

                        r = np.sqrt( np.sum((cur_pos - self._circ_center)**2) )

                        self.DMS("SELECTION", "Radius: " + str(r) + ")", level="A")

                        print "C center", self.selection_circ.center, "radius", self.selection_circ.get_radius()

                        self.get_analysis(center=self._circ_center, radius=r)
                else:

                    self._circ_center = self._circ_center + \
                        (cur_pos - self._dragging_origin)

                    self.selection_circ.center = tuple(self._circ_center)


                    self.get_analysis(center=self._circ_center, \
                        radius = self.selection_circ.get_radius())
                

    def get_img_section(self, ul, lr, as_copy=False, dtype=None):


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

        self.DMS("Analyse one", "Selection shape: {0}".format(self.f_settings.A.\
            _img[upper:lower,left:right].shape),level="D",debug_level="debug")

        if as_copy:
            if dtype is None:
                return np.copy(self.f_settings.A._img[upper:lower,left:right])
            else:
                return np.copy(self.f_settings.A._img[upper:lower,left:right].astype(dtype))

        else:
            return self.f_settings.A._img[upper:lower,left:right]

    def get_features(self, cell=None):

        if cell is not None:

            self._cell = cell

        if self._cell is None:
            return None


        if self.last_value_space == self.CELL_ESTIMATE:

            self._cell.kodak_data_source = self._cell.original_data_source.copy()

            self._cell.set_new_data_source_space(\
                space='cell estimate',
                bg_sub_source = self._cell.get_item('background').filter_array,
                polynomial_coeffs = self._cce_poly_coeffs)

            self.DMS('ANALYSE ONE', 'Cell Estimate conversion is {0}'.format(\
                not(np.all(self._cell.kodak_data_source == self._cell.data_source))),
                "L", debug_level="debug")
        else:

            if self._cell.kodak_data_source is not None:

                self._cell.data_source = self._cell.kodak_data_source.copy()
                self._cell.kodak_data_source = None
                self._cell.set_grid_array_pointers()
                self.DMS('ANALYSE ONE', 'Kodak reversal is {0}'.format(\
                    np.all(self._cell.get_item('blob').grid_array == self._cell.original_data_source)),
                    "L", debug_level="debug")
                self.DMS('ANALYSE ONE', 'Reversed to Kodak Space', "L",
                    debug_level="debug")
            else:
                self.DMS('ANALYSE ONE', 
                    'No reversal to Kodak Space needed, already there',
                    "L", debug_level="debug")

        features = self._cell.get_analysis(no_detect=True)

        return features

    def get_analysis(self, center=None, radius=None):

        if radius is None:
            self.selection_circ.set_radius(0)
            
        #EMPTYING self.plots_vbox
        #for child in self.plots_vbox2.get_children():
        #    self.plots_vbox2.remove(child)


        img_transf = self.get_kodak_image()

        if img_transf is None:
            return None

        x_factor = img_transf.shape[0] / 200
        y_factor = img_transf.shape[1] / 200

        if x_factor > y_factor:
            scale_factor = x_factor
        else:
            scale_factor = y_factor
        if scale_factor == 0:
            scale_factor = 1



        #
        # BLOB SELECTION CANVAS
        #
        image_size = (img_transf.shape[1]/scale_factor,
            img_transf.shape[0]/scale_factor)


        #
        # RETRIEVING ANALYSIS
        #

        self._cell = colonies.get_grid_cell_from_array(img_transf, center=center, radius = radius)
        self._cell.kodak_data_source = None
        self._cell.original_data_source = img_transf.copy()
        features = self.get_features() 

        self.set_features_in_gui(features)

        self.DMS("ANALYSE ONE", 'Features: {0}'.format(features),
            "L", debug_level="debug")

        #
        # UPDATE IMAGE SECTION USING CURRENT VALUE SPACE REPRESENTATION
        #

        if center is None and radius is None:

            self.blob_fig_ax.set_data(self._cell.data_source.T)

            self.blob_fig_ax.set_clim(vmin = 0, 
                vmax=(100,3500)[self.last_value_space])

            self.blob_fig.gca().add_patch(self.selection_circ)

        self.blob_fig.canvas.draw()
 
       
        #
        # GETTING BLOB AND BACKGROUND
        #

        blob = self._cell.get_item('blob')
        self.blob_filter = blob.filter_array
        self.blob_image = blob.grid_array
        background = self._cell.get_item('background')
        #
        # CALCULATING CCE IF POLY-COEFFS EXISTS
        #

        if self._cce_poly_coeffs is not None:

            cce_vector = self.get_cce_data_vector(img_section=img_transf)
            cce_vector = self.get_expanded_vector(cce_vector)
            cce_calculated = self.get_vector_polynomial_sum_single(cce_vector, self._cce_poly_coeffs)
            self.cce_calculated.set_text(str(cce_calculated) + " cells in blob")



        #
        # BLOB vs BACKGROUND CANVAS
        #


        blob_filter_view = self.blob_filter#.astype(float) * 256 + img_transf
        #blob_filter_view = blob_filter_view * ( 256/ float(np.max(blob_filter_view)) )

        self.blob_bool_fig_ax.set_data(blob_filter_view.T)
        self.blob_bool_fig.canvas.draw()


        self.bg_bool_fig_ax.set_data(background.filter_array.T)
        self.bg_bool_fig.canvas.draw()

        #
        # HISTOGRAM
        #

        blob_hist = blob.histogram
        if blob_hist.labels != None:
            bincenters = 0.5*(blob_hist.labels[1:]+blob_hist.labels[:-1])

            if self._blobs_have_been_loaded == False:
            
                label = gtk.Label("Histogram (Kodak Space):")
                label.show()
                self.plots_vbox2.pack_start(label, False, False, 2)

                self.blob_hist = plt.Figure(figsize=image_size, dpi=150)
                self.blob_hist.add_axes()
                
                image_canvas = FigureCanvas(self.blob_hist)
                image_plot = self.blob_hist.gca()

                image_canvas.set_size_request(300,400)
                image_canvas.show()
                self.plots_vbox2.pack_start(image_canvas, False, False, 2)

                #self.blob_hist.subplots_adjust(top=2, bottom=2)

                label = gtk.Label("Threshold (red), Background Mean(green)")
                label.show()
                self.plots_vbox2.pack_start(label, False, False, 2)
            else:
                image_plot = self.blob_hist.gca()
                image_plot.cla()

            #image_plot.bar(blob_hist.labels, blob_hist.counts, linewidth=0, color='k')
            image_plot.hist(img_transf.ravel(), bins=150, color='k', alpha=0.6, lw=1)
            #image_plot.hist(img_transf[np.where(background.filter_array)].ravel(), bins=150, color='k', alpha=0.6, lw=1)

            x_labels = [t.get_text() for t  in image_plot.get_axes().get_xticklabels()]
            self.DMS('ANALYSE ONE', 'Debugging niceness of plot {0}'.format(\
                [str(t) for t  in image_plot.get_axes().get_xticklabels()]), 
                "L", debug_level='debug')

            image_plot.get_axes().set_xticklabels(x_labels, fontsize='xx-small')

            #if blob_hist.labels != None:
            image_plot.axvline(np.mean(img_transf[np.where(\
                background.filter_array)]), c='g')


            #x_ticks = range(0,256,20)           
            #image_plot.set_xticks(x_ticks)
            #image_plot.set_xticklabels(map(str,x_ticks), fontsize='xx-small')
            image_plot.axvline(blob.threshold, c='r')
            #image_plot.set_xlim(xmin=0, xmax=100)


        self.blob_hist.canvas.draw()

        
        if self._blobs_have_been_loaded == False:
            self.plots_vbox2.show_all()
            self._blobs_have_been_loaded = True

