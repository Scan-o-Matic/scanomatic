#! /usr/bin/env python

# 
# colonies.py   v 0.1
#
# This is a convienience module for command line calling of all different types of colony
# analysis that are implemented.
#
# The module can also be imported directly into other scrips as a wrapper
#



#
# DEPENDENCIES
#

import matplotlib.image as plt_img
import elementtree.ElementTree as ET
import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import grid_array
import simple_conf as conf

#
# FUNCTIONS
#

def print_progress_bar(fraction, size=40):
    prog_str = "["
    fraction *= size
    for i in xrange(size):
        if fraction > i:
            prog_str += "="
        else:
            prog_str += " "

    prog_str += "]"

    print 
    print
    print prog_str
    print
    print

def analyse_project(log_file_path, outdata_file_path, pinning_matrices, \
        verboise=False, use_fallback = False, use_otsu = False):
    """
        analyse_project parses a log-file and runs a full analysis on all 
        images in it. It will step backwards in time, starting with the 
        last image.

        The function takes the following arguments:

        @log_file_path      The path to the log-file to be processed

        @outdata_file_path  The path to the file were the analysis will
                            be put

        @pinning_matrices   A list/tuple of (row, columns) pinning 
                            matrices used for each plate position 
                            respectively

        @verboise           Will print some basic output of progress.

        @use_fallback       Determines if fallback colony detection
                            should be used.

        @use_otsu           Determines if Otsu-thresholding should be
                            used when looking finding the grid (not
                            default, and should not be used)
        
        The function returns None if nothing was done of if it crashed.
        If it runs through it returns the number of images processed.

    """
    
    log_file = conf.Config_File(log_file_path)

    project_image = Project_Image(pinning_matrices)

    image_dictionaries = log_file.get_all("%n")
    if image_dictionaries == None:
        return None

    plate_position_keys = []
    plates = len(pinning_matrices)

    for i in xrange(plates):
        plate_position_keys.append("plate_" + str(i) + "_area")

    image_pos = len(image_dictionaries) - 1
    image_tot = image_pos 

    if verboise:
        print "*** Project has " + str(image_pos+1) + " images."
        print "* Nothing to wait for"
        print

    ET_root = ET.Element("project")
    ET_start = ET.SubElement(ET_root, "start-time")
    ET_start.text = str(image_dictionaries[0]['Time'])
    ET_scans = ET.SubElement(ET_root, "scans")

    while image_pos >= 0:

        img_dict_pointer = image_dictionaries[image_pos]

        plate_positions = []

        for i in xrange(plates):
            plate_positions.append( \
                img_dict_pointer[plate_position_keys[i]] )

        if verboise:
            print "*** Analysing: " + str(img_dict_pointer['File'])
            print

        features = project_image.get_analysis( img_dict_pointer['File'], \
            plate_positions, img_dict_pointer['grayscale_values'], use_fallback, use_otsu )

        if verboise:
            print "*** Building report"
            print

        ET_scan = ET.SubElement(ET_scans, "scan")
        ET_scan.set("index", str(image_pos))

        ET_scan_valid = ET.SubElement(ET_scan,"scan-valid")

        if features != None:
            ET_scan_valid.text = str(1)

            ET_scan_gs_calibration = ET.SubElement(ET_scan, "calibration")
            ET_scan_gs_calibration.text = \
                str(img_dict_pointer['grayscale_values'])
            ET_scan_time = ET.SubElement(ET_scan, "time")
            ET_scan_time.text = str(img_dict_pointer['Time'])
            ET_plates = ET.SubElement(ET_scan, "plates")
            for i in xrange(plates):
                ET_plate = ET.SubElement(ET_plates, "plate")
                ET_plate.set("index", str(i))

                ET_plate_matrix = ET.SubElement(ET_plate, "plate-matrix")
                ET_plate_matrix.text = str(pinning_matrices[i])

                ET_plate_R = ET.SubElement(ET_plate, "R")
                ET_plate_R.text = str(project_image.R[i])

                ET_cells = ET.SubElement(ET_plate, "cells")

                for x, rows in enumerate(features[i]):
                    for y, cell in enumerate(rows):

                        ET_cell = ET.SubElement(ET_cells, "cell")
                        ET_cell.set("x", str(x))
                        ET_cell.set("y", str(y))

                        if cell != None:
                            for item in cell.keys():

                                ET_cell_item = ET.SubElement(ET_cell, item)
                                
                                for measure in cell[item].keys():

                                    ET_measure = ET.SubElement(ET_cell_item,\
                                        measure)

                                    ET_measure.text = str(cell[item][measure])


        else:
            ET_scan_valid.text = str(0)

        image_pos -= 1

        #DEBUGHACK
        #if image_pos > 1:
        #    image_pos = 1 
        #DEBUGHACK - END


        if verboise:
            print_progress_bar((image_tot-image_pos)/float(image_tot), size=70)


    tree = ET.ElementTree(ET_root)
    tree.write(outdata_file_path)

#
# CLASS Project_Image
#

class Project_Image():
    def __init__(self, pinning_matrices, im_path=None, plate_positions=None ):

        self._im_path = im_path
        self._im_loaded = False
        
        self._plate_positions = plate_positions
        self._pinning_matrices = pinning_matrices

        self.im = None
        self._grid_arrays = []
        self.features = []
        self.R = []

        for a in xrange(len(pinning_matrices)):
            self._grid_arrays.append(grid_array.Grid_Array(pinning_matrices[a]))
            self.features.append(None)
            self.R.append(None)

    def get_analysis(self, im_path, features, grayscale_values, \
            use_fallback=False, use_otsu=True):

        if im_path != None:
            self._im_path = im_path

        try:
            self.im = plt_img.imread(self._im_path)
            self._im_loaded = True
        except:
            print "*** Error: Could not open image at " + str(self._im_path)
            self._im_loaded = False

        if self._im_loaded == True:           
            if len(self.im.shape) > 2:
                self.im = self.im[:,:,0]
        else:
            return None

        if len(grayscale_values) > 3:
            gs_values = np.array(grayscale_values)
            gs_indices = np.arange(len(grayscale_values))

            gs_fit = np.polyfit(gs_indices, gs_values,3)
        else:
            gs_fit = None

        scale_factor = 4.0

        for grid_array in xrange(len(self._grid_arrays)):

            x0 = int(features[grid_array][0][0]*scale_factor)
            x1 = int(features[grid_array][1][0]*scale_factor)
            if x0 < x1:
                upper = x0
                lower = x1
            else:
                upper = x1
                lower = x0

            y0 = int(features[grid_array][0][1]*scale_factor)
            y1 = int(features[grid_array][1][1]*scale_factor)
            if y0 < y1:
                left = y0
                right = y1
            else:
                left = y1
                right = y0

            self._grid_arrays[grid_array].get_analysis( \
                self.im[ upper:lower, left:right ], \
                gs_values=gs_values, use_fallback=use_fallback, use_otsu=use_otsu, median_coeff=None, \
                verboise=False, visual=False)

            self.features[grid_array] = self._grid_arrays[grid_array]._features
            self.R[grid_array] = self._grid_arrays[grid_array].R

        return self.features


if __name__ == "__main__":

    import sys

    print "*** This IS a test ***"
    print

    if len(sys.argv) != 3:

        print "COMMAND: ", sys.argv[0], "[log-file] [analysis-xml-file-output]"
        
    else:
        pm = [(16,24), (16,24), (16,24), (16,24)]
        
        #last is if Otsu, second last is for fallback
        analyse_project(sys.argv[1], sys.argv[2],pm,True, True, False)
