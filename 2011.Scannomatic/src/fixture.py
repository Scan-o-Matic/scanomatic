#!/usr/bin/env python
"""GTK-GUI for setting up a fixture"""

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

from PIL import Image, ImageWin

import pygtk
pygtk.require('2.0')

import gtk, pango
import os, os.path, sys
import types

import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import src.settings_tools as settings_tools

class Fixture_GUI(gtk.Frame):

    def __init__(self, owner):

        gtk.Frame.__init__(self, "CONFIGURATION OF FIXTURES")

        self.owner = owner
        #configuring a fixture
        self.current_fixture_settings = []
        self._fixture_gui_updating = False
        self.fixture_active_pos = None
        self.fixture_active_pos_setting = None

        self._fixture_config_root = self.owner._program_config_root + os.sep + "fixtures"
        self.f_settings = settings_tools.Fixture_Settings(self._fixture_config_root, fixture="fixture_a")
        analysis_img = self._fixture_config_root + os.sep + self.f_settings.fixture_name + ".tiff"

        vbox2 = gtk.VBox(False, 0)
        vbox2.show()
        self.add(vbox2)

        label = gtk.Label("This is in development...please bare with me")
        label.show()
        vbox2.pack_start(label, False, False, 10)

        hbox = gtk.HBox()
        hbox.show()
        vbox2.pack_start(hbox, False, False, 0)

        if os.path.isfile(analysis_img):
            self.fixture_analysis_image = plt_img.imread(analysis_img)

        scale_factor = 2
        image_size = (self.fixture_analysis_image.shape[0]/scale_factor,
            self.fixture_analysis_image.shape[1]/scale_factor)

        self.plot_selecting = False
        self.plot_ul  = (0,0)

        self.owner.DMS("Fixture init","Setting up figure display",100)
        self.image_fig = plt.Figure(figsize=image_size, dpi=150)
        self.image_plot = self.image_fig.add_subplot(111)
        self.image_canvas = FigureCanvas(self.image_fig)
        self.owner.DMS("Fixture init","Figure: Connetcting events",100)
        self.image_fig.canvas.mpl_connect('button_press_event', self.plot_click)
        self.image_fig.canvas.mpl_connect('button_release_event', self.plot_release)
        self.image_fig.canvas.mpl_connect('motion_notify_event', self.plot_move)

        self.owner.DMS("Fixture init","Figure: Plotting image",100)
        self.image_ax = self.image_plot.imshow(self.fixture_analysis_image)

        self.owner.DMS("Fixture init","Figure: Initialising selection rectangle",100)
        self.selection_rect = plt_patches.Rectangle(
                (0,0),0,0, ec = 'k', fill=False, lw=0.2
                )
        self.selection_rect.get_axes()
        self.selection_rect.get_transform()
        self.image_plot.add_patch(self.selection_rect)

#        self.image_plot.canvas.draw()
        self.image_plot.get_xaxis().set_visible(False)
        self.image_plot.get_yaxis().set_visible(False)
        self.image_plot.set_xlim(xmin=0,xmax=self.fixture_analysis_image.shape[1])
        self.image_plot.set_ylim(ymin=0,ymax=self.fixture_analysis_image.shape[0])
        self.image_canvas.show()
        self.image_canvas.set_size_request(image_size[1], image_size[0])
        hbox.pack_start(self.image_canvas, False, False, 2)
        
        vbox3 = gtk.VBox(False, 0)
        vbox3.show()
        hbox.pack_end(vbox3, False, False, 0)

        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Select image of fixture:")
        label.show()
        hbox.pack_start(label, False, False, 2)

        button = gtk.Button(label = 'Open')
        button.show()
        button.connect("clicked", self.select_image)
        hbox.pack_end(button, False, False, 2)

        self.fixture_image = gtk.Label("")
        self.fixture_image.set_max_width_chars(60)
        self.fixture_image.set_ellipsize(pango.ELLIPSIZE_START)
        self.fixture_image.show()
        vbox3.pack_start(self.fixture_image, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Select fixture marking image:")
        label.show()
        hbox.pack_start(label, False, False, 2)

        button = gtk.Button(label = 'Open')
        button.show()
        button.connect("clicked", self.select_marking)
        hbox.pack_end(button, False, False, 2)

        self.fixture_marking = gtk.Label("")
        self.fixture_marking.set_max_width_chars(50)
        self.fixture_marking.set_ellipsize(pango.ELLIPSIZE_MIDDLE)
        self.fixture_marking.show()
        vbox3.pack_start(self.fixture_marking, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label = gtk.Label("Number of markings:")
        label.show()
        hbox.pack_start(label, False, False, 2)

        self.fixture_markings_count = gtk.Entry()
        self.fixture_markings_count.show()
        self.fixture_markings_count.set_text("")
        self.fixture_markings_count.connect("focus-out-event", self.verify_input, "int")
        hbox.pack_end(self.fixture_markings_count, False, False, 2)

        self.fixture_analysis_button = gtk.Button(label = 'Run analysis of markings placement')
        self.fixture_analysis_button.connect("clicked", self.marker_analysis)    
        self.fixture_analysis_button.show()
        vbox3.pack_start(self.fixture_analysis_button, False, False, 2)

        self.fixture_grayscale_checkbox = gtk.CheckButton(label="Grayscale", use_underline=False)
        self.fixture_grayscale_checkbox.show()
        self.fixture_grayscale_checkbox.connect("clicked", self.areas_update)
        vbox3.pack_start(self.fixture_grayscale_checkbox, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox3.pack_start(hbox, False, False, 2)

        label= gtk.Label("Plate Positions:")
        label.show()
        hbox.pack_start(label, False, False, 2)
 
        self.fixture_platepositions = gtk.Entry()
        self.fixture_platepositions.show()
        self.fixture_platepositions.set_text("4")
        self.fixture_platepositions.connect("focus-out-event", self.areas_update, "int")
        hbox.pack_end(self.fixture_platepositions, False, False, 2)

        frame = gtk.Frame("Area config")
        frame.show()
        vbox3.pack_start(frame, False, False, 2)

        vbox4 = gtk.VBox()
        vbox4.show()
        frame.add(vbox4)

        self.fixture_area_selection = gtk.combo_box_new_text()
        self.fixture_area_selection.show()
        self.fixture_area_selection.connect("changed", self.areas_update)
        vbox4.pack_start(self.fixture_area_selection, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox4.pack_start(hbox, False, False, 2)

        self.fixture_area_ul_button = gtk.ToggleButton(label ='Select upper left corner')
        self.fixture_area_ul_button.connect("clicked", self.area_pos,"ul")
        #self.fixture_area_ul_button.show()
        hbox.pack_start(self.fixture_area_ul_button, False, False, 2)

        self.fixture_area_ul = gtk.Label("")
        self.fixture_area_ul.show()
        hbox.pack_end(self.fixture_area_ul, False, False, 2)

        hbox = gtk.HBox()
        hbox.show()
        vbox4.pack_start(hbox, False, False, 2)

        self.fixture_area_lr_button = gtk.ToggleButton(label = 'Select lower right corner')
        self.fixture_area_lr_button.connect("clicked", self.area_pos,"lr")
        #self.fixture_area_lr_button.show()
        hbox.pack_start(self.fixture_area_lr_button, False, False, 2)

        self.fixture_area_lr = gtk.Label("")
        self.fixture_area_lr.show()
        hbox.pack_end(self.fixture_area_lr, False, False, 2)

        button = gtk.Button(label='Save fixture settings')
        button.show()
        button.connect("clicked", self.config_file_save)
        vbox3.pack_start(button, False, False, 2)

        self.areas_update()

    def plot_click(self, event=None):

        self.plot_ul = (event.xdata, event.ydata)
        self.selection_rect.set_x(event.xdata)
        self.selection_rect.set_y(event.ydata)
        self.plot_selecting = True

    def plot_move(self, event=None):

        if self.plot_selecting and event.xdata != None and event.ydata != None:
            #self.selection_rect = plt_patches.Rectangle(
                #self.plot_ul , 
            self.selection_rect.set_width(    event.xdata - self.plot_ul[0])#,
            self.selection_rect.set_height(    event.ydata - self.plot_ul[1])#,
                #ec = 'k', fc='b', fill=True, lw=1,
                #axes = self.image_ax)
            self.image_fig.canvas.draw() 
            self.owner.DMS("SELECTING", "Selecting something in the image", 1)

    def plot_release(self, event=None):

        self.plot_selecting = False

        if event.xdata == None or event.ydata == None:
            self.selection_rect.set_width( 0 )
            self.selection_rect.set_height( 0 )
            self.image_fig.canvas.draw()

        else:
            self.owner.DMS("SELECTION", "UL: " + str(self.plot_ul) + ", LR: (" + 
                str(event.xdata) + ", "  +
                str(event.ydata) + ")", level=1)

                #self.fixture_active_pos(str((x_pos, y_pos)))
            lr = (event.xdata, event.ydata)
            area = [self.plot_ul, lr] 
            self.f_settings.fixture_config_file.set(self.settings_name(), area)
            self.fixture_area_ul.set_text( str(area[0]) )
            self.fixture_area_lr.set_text( str(area[1]) )


    def config_file_save(self, widget=None, event=None, data=None):
        self.f_settings.fixture_config_file.save()


    def settings_name(self, active = None):
            if active == None:
                active = self.fixture_area_selection.get_active()

            if active >= 1:
                active -= 1
                if self.fixture_grayscale_checkbox.get_active():
                    active -= 1 
                if active >= 0:
                    value_name = "plate_" + str(active)
                else:
                    value_name = "grayscale"

                value_name += "_area"

                return value_name
            else:
                return None

    def image_click(self, widget=None, event=None, data=None):
        vertOffset = float( self.fixture_scrolled_window.get_vadjustment().value )
        horzOffset = float( self.fixture_scrolled_window.get_hadjustment().value )
        
        if event:
            x_pos = (event.x + horzOffset)
            y_pos = (event.y + vertOffset)
            #print "x:", horzOffset, event.x, widget.get_allocation().x
            #print "y:", vertOffset, event.y, widget.get_allocation().y
        if self.fixture_active_pos != None and self.fixture_active_pos_setting != None:
            self.fixture_active_pos(str((x_pos, y_pos)))
            area = [eval(self.fixture_area_ul.get_text()), eval(self.fixture_area_lr.get_text())] 
            self.f_settings.fixture_config_file.set(self.settings_name(), area)

        else:
            self.owner.DMS("Clicked Image", "Pos: " + str((x_pos, y_pos)), level=1)

    def area_pos(self, widget=None, event=None, data=None):
        if self._fixture_gui_updating == False:
            self._fixture_gui_updating = True 
            if event == "ul":
                self.fixture_active_pos = self.fixture_area_ul.set_text
                self.fixture_area_lr_button.set_active(False)
                self.fixture_active_pos_setting = 0
            elif event == "lr":
                self.fixture_active_pos = self.fixture_area_lr.set_text
                self.fixture_area_ul_button.set_active(False)
                self.fixture_active_pos_setting = 1 
            else:
                self.fixture_active_pos = None
                self.fixture_area_ul_button.set_active(False)
                self.fixture_area_lr_button.set_active(False)
                self.fixture_active_pos_setting = None
            self._fixture_gui_updating = False

    def marker_analysis(self, widget=None, event=None, data=None):
        analysis_img = self.f_settings.marker_analysis(fixture_setup=True)
        if analysis_img != None:
            self.fixture_analysis_image.set_from_file(analysis_img)
            self.fixture_scrolled_window.size_request()
        else:
            self.owner.DMS("Error", "Image anaylsis failed", level = 1010)

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
            self.fixture_image.set_text(newimg.get_filename())
            self.f_settings.image_path = newimg.get_filename()

        newimg.destroy()

    def select_marking(self, widget=None, event=None, data=None):
        newimg = gtk.FileChooserDialog(title="Select marking image", action=gtk.FILE_CHOOSER_ACTION_OPEN, 
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

        f = gtk.FileFilter()
        f.set_name("Valid image files")
        f.add_mime_type("image/png")
        f.add_pattern("*.png")
        f.add_pattern("*.PNG")
        newimg.add_filter(f)

        result = newimg.run()
        
        if result == gtk.RESPONSE_APPLY:
            self.fixture_marking.set_text(newimg.get_filename())
            self.f_settings.marking_path = newimg.get_filename()
            try:
                self.f_settings.markings = int(self.fixture_markings_count.get_text())
            except ValueError:
                self.f_settings.markings = 0
 
            self.f_settings.fixture_config_file.set("marker_path",self.fixture_marking.get_text())
            self.f_settings.fixture_config_file.set("marker_count", self.f_settings.markings)
 
        newimg.destroy()

    def verify_input(self, widget=None, event=None, data=None):
        if widget:
            input_to_test = widget.get_text()        
            if data == "int":
                try:
                    int(input_to_test)
                    return True
                except ValueError:
                    widget.set_text("")
                    return False

            #HACK
            self.f_settings.markings = int(self.fixture_markings_count.get_text())
            self.f_settings.fixture_config_file.set("marker_count", self.f_settings.markings)
                    
    def areas_update(self, widget=None, event=None, data=None):
        if self._fixture_gui_updating == False:
            self._fixture_gui_updating = True
            if widget:
                if data == "int":
                    self.verify_input(widget=widget, event=event, data=data)
            elif widget == None:
                self.fixture_marking.set_text(str(self.f_settings.fixture_config_file.get("marker_path")))
                self.fixture_markings_count.set_text(str(self.f_settings.fixture_config_file.get("marker_count")))                
                try:
                    self.fixture_grayscale_checkbox.set_active(self.f_settings.fixture_config_file.get("grayscale"))
                except TypeError:
                    self.fixture_grayscale_checkbox.set_active(False)

            if self.fixture_grayscale_checkbox.get_active():
                grayscales = 1
                self.f_settings.fixture_config_file.set("grayscale", True)
            else:
                grayscales = 0
                self.f_settings.fixture_config_file.set("grayscale", False)
                self.f_settings.fixture_config_file.delete("grayscale_area")

            if self.fixture_platepositions.get_text() != "":
                plates = int(self.fixture_platepositions.get_text())
            else:
                plates = 0

            plates += grayscales

            if self.current_fixture_settings != None:
                if len(self.current_fixture_settings) > plates:
                    for i in range(plates, len(self.current_fixture_settings)):
                        self.f_settings.fixture_config_file.delete(
                            self.f_settings.settings_name(i))

            #A BIT UGLY
            if grayscales > 0:
                self.f_settings.fixture_config_file.set('grayscale_area',[(-1,-1),(-1,-1)], overwrite=False)

            if plates > grayscales:
                for p in xrange(plates-grayscales):
                    self.f_settings.fixture_config_file.set('plate_'+str(p)+'_area',[(-1,-1),(-1,-1)], overwrite=False)

            active = self.fixture_area_selection.get_active()
            i = 0 #HACK!!
            while i < 15:
                self.fixture_area_selection.remove_text(0)
                i += 1

            self.fixture_area_selection.append_text("Select an area")
            self.current_fixture_settings = []

            for i in xrange(plates):
                if i == 0 and grayscales == 1:
                    self.fixture_area_selection.append_text("Grayscale")
                else:
                    self.fixture_area_selection.append_text("Plate " + str(i-grayscales))
                self.current_fixture_settings.append(" ")

            if active < 0:
                active = 0
            else:
                if active <= len(self.current_fixture_settings):
                    self.fixture_area_selection.set_active(active)
                else:
                    self.fixture_area_selection.set_active(0)

            active_name = self.settings_name()

            
            if active_name != None:
                cur_area = self.f_settings.fixture_config_file.get(active_name)
                if cur_area != None:
                    self.fixture_area_ul.set_text(str(cur_area[0]))
                    self.fixture_area_lr.set_text(str(cur_area[1]))
            else:
                self.fixture_area_ul.set_text("(N/A, N/A)")
                self.fixture_area_lr.set_text("(N/A, N/A)")
                self.fixture_area_selection.set_active(0)

            self.fixture_area_selection.set_sensitive(len(self.current_fixture_settings) > 0)
            self.fixture_area_ul_button.set_sensitive(self.fixture_area_selection.get_active()>0)
            self.fixture_area_lr_button.set_sensitive(self.fixture_area_selection.get_active()>0)

            self._fixture_gui_updating = False

        return False

