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

import matplotlib
#matplotlib.use('Agg')
import matplotlib.image as plt_img
#import elementtree.ElementTree as ET
import types
import numpy as np
from time import time
from argparse import ArgumentParser #, FileType

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
    printmeon = True
    print

def analyse_project(log_file_path, outdata_file_path, pinning_matrices, \
        graph_watch, graph_output, supress_analysis = False, \
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

        @graph_watch        A coordinate PLATE:COL:ROW for a colony to
                            watch particularly.

        @graph_output       An optional PATH to where to save the graph
                            produced by graph_watch being set.

        @supress_analysis  Suppresses the main analysis and thus
                            only graph_watch thing is produced.

        @verboise           Will print some basic output of progress.

        @use_fallback       Determines if fallback colony detection
                            should be used.

        @use_otsu           Determines if Otsu-thresholding should be
                            used when looking finding the grid (not
                            default, and should not be used)
        
        The function returns None if nothing was done of if it crashed.
        If it runs through it returns the number of images processed.

    """
    
    start_time = time()


    if graph_watch != None:
        from matplotlib import pyplot
        from matplotlib.font_manager import FontProperties
        from PIL import Image
        watch_reading = [] 

        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt_watch_colony = pyplot.Figure()
        plt_watch_1 = plt_watch_colony.add_subplot(411)
        pict_target_width = 40
        plt_watch_1.axis((0, (pict_target_width + 1) * 217, 0, pict_target_width * 3), frameon=False,
            title='Plate: ' + str(graph_watch[0]) + ', position: (' + str(graph_watch[1]) + ', ' + str(graph_watch[2]) + ')')
        plot_labels = []

    log_file = conf.Config_File(log_file_path)


    image_dictionaries = log_file.get_all("%n")
    if image_dictionaries == None:
        return None

    plate_position_keys = []
    plates = len(pinning_matrices)


    for i in xrange(plates):
        if supress_analysis != True or graph_watch[0] == i:
            plate_position_keys.append("plate_" + str(i) + "_area")

    if supress_analysis == True:
        project_image = Project_Image([pinning_matrices[graph_watch[0]]])
        graph_watch[0] = 0
        plates = 1
    else:
        try:
            fh = open(outdata_file_path,'w')
        except:
            print "*** Error, can't open target file:", outdata_file_path
            return False
        description = None
        interval_time = None
        project_image = Project_Image(pinning_matrices)

    image_pos = len(image_dictionaries) - 1
    image_tot = image_pos 

    if verboise:
        print "*** Project has " + str(image_pos+1) + " images."
        print "* Nothing to wait for"
        print

    if supress_analysis != True:
        fh.write('<project>')
        #ET_root = ET.Element("project")

        fh.write('<start-time>' + str(image_dictionaries[0]['Time']) + '</start-time>')
        #ET_start = ET.SubElement(ET_root, "start-time")
        #ET_start.text = str(image_dictionaries[0]['Time'])

        fh.write('<description>' + str(description) + '</description>')
        #ET_description = ET.SubElement(ET_root, "description")
        #ET_description.text = "Placeholder description"

        fh.write('<number-of-scans>' + str(image_pos+1) + '</number-of-scans>')
        #ET_number_of_scans = ET.SubElement(ET_root, "number-of-scans")
        #ET_number_of_scans.text = str(image_pos+1)

        fh.write('<interval-time>' + str(interval_time) + '</interval-time>')
        #ET_interval = ET.SubElement(ET_root, "interval-time")
        #ET_interval.text = "Placeholder"

        fh.write('<plates-per-scan>' + str(plates) + '</plates-per-scan>')
        #ET_plates_per_scan = ET.SubElement(ET_root, "plates-per-scan")
        #ET_plates_per_scan.text = str(plates)

        fh.write('<pinning-matrices>')
        #ET_pinning_matrices = ET.SubElement(ET_root, "pinning-matrices")
        for pos in xrange(plates):
            fh.write('<pinning-matrix index="' + str(pos) + '">' + \
                 str(pinning_matrices[pos]) + '</pinning-matrix>')
            #ET_matrix = ET.SubElement(ET_pinning_matrices, "pinning-matrix")
            #ET_matrix.set("index", str(pos))
            #ET_matrix.text = str(pinning_matrices[pos])

        fh.write('</pinning-matrices>')

        fh.write('<scans>')
        #ET_scans = ET.SubElement(ET_root, "scans")

    while image_pos >= 0:
        scan_start_time = time()
        img_dict_pointer = image_dictionaries[image_pos]

        plate_positions = []

        for i in xrange(plates):
            plate_positions.append( \
                img_dict_pointer[plate_position_keys[i]] )

            if verboise:
                print "** Position", plate_position_keys[i], ":", img_dict_pointer[plate_position_keys[i]]

        if verboise:
            print
            print "*** Analysing: " + str(img_dict_pointer['File'])
            print

        features = project_image.get_analysis( img_dict_pointer['File'], \
            plate_positions, img_dict_pointer['grayscale_values'], use_fallback, use_otsu, watch_colony=graph_watch, supress_other=supress_analysis )

        if supress_analysis != True:
            fh.write('<scan index="' + str(image_pos) + '">')
            #ET_scan = ET.SubElement(ET_scans, "scan")
            #ET_scan.set("index", str(image_pos))

            fh.write('<scan-valid>')
            #ET_scan_valid = ET.SubElement(ET_scan,"scan-valid")

        if features == None:
            if supress_analysis != True:
                fh.write(str(0) + '</scan-valid>')
                #ET_scan_valid.text = str(0)
        else:
            if graph_watch != None:

                pict_size = project_image.watch_grid_size
                pict_scale = pict_target_width / float(pict_size[1])
                pict_resize = (int(pict_size[0] * pict_scale), int(pict_size[1] * pict_scale))


                plt_watch_1.imshow(Image.fromstring('L', (project_image.watch_scaled.shape[1], \
                    project_image.watch_scaled.shape[0]), \
                    project_image.watch_scaled.tostring()).resize(pict_resize, Image.BICUBIC), \
                    extent=(image_pos * pict_target_width, (image_pos+1)*pict_target_width-1, 10, 10 + pict_resize[1]))


                plt_watch_1.imshow(Image.fromstring('L', (project_image.watch_blob.shape[1], \
                    project_image.watch_blob.shape[0]), \
                    project_image.watch_blob.tostring()).resize(pict_resize, Image.BICUBIC), \
                    extent=(image_pos * (pict_target_width), (image_pos+1)*(pict_target_width)-1, 10 + pict_resize[1] + 1, 10 + 2 * pict_resize[1] + 1))



                #project_image.watch_blob.shape
                tmp_results = []
                for cell_item in project_image.watch_results.keys():
                    for measure in project_image.watch_results[cell_item].keys():
                        tmp_results.append(project_image.watch_results[cell_item][measure])
                        if len(watch_reading) == 0:
                            plot_labels.append(cell_item + ':' + measure)
                watch_reading.append(tmp_results)    
                
                #HACK START: DEBUGGING
                print "*** R:", project_image.R
                #if image_pos < 200:
                #    image_pos = -1 
                #HACK END

            if supress_analysis != True:
                fh.write(str(1) + '</scan-valid>')
                #ET_scan_valid.text = str(1)

                fh.write('<calibration>' + str(img_dict_pointer['grayscale_values']) + '</calibration>')
                #ET_scan_gs_calibration = ET.SubElement(ET_scan, "calibration")
                #ET_scan_gs_calibration.text = \
                #    str(img_dict_pointer['grayscale_values'])

                fh.write('<time>' + str(img_dict_pointer['Time']) + '</time>')
                #ET_scan_time = ET.SubElement(ET_scan, "time")
                #ET_scan_time.text = str(img_dict_pointer['Time'])

                fh.write('<plates>')
                #ET_plates = ET.SubElement(ET_scan, "plates")

                for i in xrange(plates):
                    fh.write('<plate index="' + str(i) + '">')
                    #ET_plate = ET.SubElement(ET_plates, "plate")
                    #ET_plate.set("index", str(i))

                    fh.write('<plate-matrix>' + str(pinning_matrices[i]) + '</plate-matrix>')
                    #ET_plate_matrix = ET.SubElement(ET_plate, "plate-matrix")
                    #ET_plate_matrix.text = str(pinning_matrices[i])

                    fh.write('<R>' + str(project_image.R[i]) + '</R>')
                    #ET_plate_R = ET.SubElement(ET_plate, "R")
                    #ET_plate_R.text = str(project_image.R[i])

                    fh.write('<grid-cells>')
                    #ET_cells = ET.SubElement(ET_plate, "cells")

                    for x, rows in enumerate(features[i]):
                        for y, cell in enumerate(rows):

                            fh.write('<grid-cell x="' + str(x) + '" y="' + str(y) + '">')
                            #ET_cell = ET.SubElement(ET_cells, "cell")
                            #ET_cell.set("x", str(x))
                            #ET_cell.set("y", str(y))

                            if cell != None:
                                for item in cell.keys():

                                    fh.write('<' + str(item) + '>')
                                    #ET_cell_item = ET.SubElement(ET_cell, item)
                                    
                                    for measure in cell[item].keys():

                                        fh.write('<' + str(measure) + '>' + \
                                            str(cell[item][measure]) + \
                                            '</' + str(measure) + '>')
                                        #ET_measure = ET.SubElement(ET_cell_item,\
                                        #    measure)

                                        #ET_measure.text = str(cell[item][measure])

                                    fh.write('</' + str(item) + '>')
                            fh.write('</grid-cell>')
                    fh.write('</grid-cells>')
                    fh.write('</plate>')
                fh.write('</plates>')

        image_pos -= 1

        #DEBUGHACK
        #if image_pos > 1:
        #    image_pos = 1 
        #DEBUGHACK - END


        if verboise:
            print "Image took", time() - scan_start_time,"seconds."
            print_progress_bar((image_tot-image_pos)/float(image_tot), size=70)

    if supress_analysis != True:
        fh.write('</scans>')
        fh.write('</project>')
        fh.close()

    if  graph_watch != None:
        Y = np.asarray(watch_reading)
        X = (np.arange(0, len(image_dictionaries))+0.5)*pict_target_width
        #graphcolors='rgbycmk'
        cur_plt_graph = ""
        plt_graph_i = 1
        for i in xrange(int(Y.shape[1])):
            if type(Y[0,i]) != np.ndarray and type(Y[0,i]) != types.ListType and type(Y[0,i]) != types.TupleType:
                try:
                    if Y[:,i].max() == Y[:,i].min():
                        scale_factor = 0
                    else:
                        scale_factor =  100 / float(Y[:,i].max() - Y[:,i].min())

                    sub_term = float(Y[:,i].min())

                    if cur_plt_graph != plot_labels[i].split(":")[0]:
                        cur_plt_graph = plot_labels[i].split(":")[0]
                        if plt_graph_i > 1:
                            plt_watch_curves.legend(loc=1, ncol=5, prop=fontP, bbox_to_anchor = (1.0, -1.0))
                        plt_graph_i += 1
                        plt_watch_curves = plt_watch_colony.add_subplot(410 + plt_graph_i,
                            title=cur_plt_graph)
 
                    plt_watch_curves.plot(X, (Y[:,i] - sub_term) * scale_factor, #+ \
                        #3*(pict_target_width+2)*(1+(i%(len(graphcolors)-1))) +\
                        #16,\
                        #graphcolors[i%len(graphcolors)] + '-',\
                        label=plot_labels[i][len(cur_plt_graph)+1:])

                except TypeError:
                    print "*** Error on", plot_labels[i], "because of something"

        plt_watch_curves.legend(loc=1, ncol=5, prop=fontP, bbox_to_anchor = (1.0, -1.0))
        if graph_output != None:
            try:
                plt_watch_colony.savefig(graph_output, dpi=300)
            except:
                pyplot.show()
        else: 
            pyplot.show()
        return False
    #if verboise:
    #    print "*** Building report"
    #    print

    #tree = ET.ElementTree(ET_root)
    #tree.write(outdata_file_path)
    print "Full analysis took", (time() - start_time)/60, "minutes"
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
            use_fallback=False, use_otsu=True, watch_colony=None, supress_other=False):

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
                verboise=False, visual=False, watch_colony=watch_colony, supress_other=supress_other)

            self.features[grid_array] = self._grid_arrays[grid_array]._features
            self.R[grid_array] = self._grid_arrays[grid_array].R

        if watch_colony != None:
            self.watch_grid_size = self._grid_arrays[watch_colony[0]]._grid_cell_size
            self.watch_source = self._grid_arrays[watch_colony[0]].watch_source
            self.watch_scaled = self._grid_arrays[watch_colony[0]].watch_scaled
            self.watch_blob = self._grid_arrays[watch_colony[0]].watch_blob
            self.watch_results = self._grid_arrays[watch_colony[0]].watch_results
        return self.features


if __name__ == "__main__":

    parser = ArgumentParser(description='The analysis script runs through a log-file (which is created when a project is run). It creates a XML-file that holds the result of the analysis')

    parser.add_argument("-i", "--input-file", type=str, dest="inputfile", help="Log-file to be parsed", metavar="PATH")
    parser.add_argument("-o", "--ouput-file", type=str, dest="outputfile", help="Path to where the XML-file should be written", metavar="PATH")

    parser.add_argument("-p", "--plates", default=4, type=int, dest="plates", help="The number of plates in the fixture", metavar="N")
    parser.add_argument("-m", "--matrices", dest="matrices", help="The pinning matrices for each plate position in the order set by the fixture config file.", metavar="(X,Y):(X,Y)...(X,Y)")

    parser.add_argument("-w", "--watch-position", dest="graph_watch", help="The position of a colony to track.", metavar="PLATE:X:Y", type=str)

    parser.add_argument("-g", "--graph-output", dest="graph_output", help="If specified the graph is not shown to the user but instead saved to taget position", type=str)

    parser.add_argument("-s", "--supress-analysis", dest="supress", default=False, type=bool, help="If set to True, main analysis will be by-passed and only the plate and position that was specified by the -w flag will be analysed and reported. That is, this flag voids the -o flag.")
    args = parser.parse_args()

    if args.matrices == None:
     
        pm = [(16,24), (16,24), (16,24), (16,24)]

    else:

        pm = args.matrices.split(':')
        pm = map(eval, pm)

    if args.inputfile == None or (args.outputfile == None and args.supress == False):
        parser.error("You need to specify both input and output file!")

    if args.graph_watch != None:
        args.graph_watch = args.graph_watch.split(":")
        try:
            args.graph_watch = map(int, args.graph_watch)
        except:
            parser.error('The watched colony could not be resolved, make sure that you follow syntax')

        if len(args.graph_watch) <> 3:
            parser.error('Bad specification of watched colony')

        if args.graph_watch[0] < args.plates:
            if not(0 <= args.graph_watch[1] <= pm[args.graph_watch[0]][0] and 0 <= args.graph_watch[2] <= pm[args.graph_watch[0]][1]):
                parser.error('The watched colony position is out of bounds (range: (0, 0)) - ' + str(pm[args.graph_watch[1]]) + ').')
        else:
            parser.error('The watched colony position has a plate number that is too high (max: ' + str(args.plates-1) + ').')

    try:
        fh = open(args.inputfile,'r')
    except:
        parser.error('Cannot open input file, please check your path...')

    fh.close()

    if args.outputfile != None:
        try:
            fh = open(args.outputfile, 'w')
        except:
            parser.error('Cannot create the output file')
        fh.close()


    if args.graph_output != None:
        try:
            fh = open(args.graph_output, 'w')
        except:
            parser.error('Cannot create the save-file for the watched colony.')

        fh.close()

    if len(pm) == args.plates:    
        analyse_project(args.inputfile, args.outputfile, pm, args.graph_watch, args.graph_output, args.supress, True, True, False)
    else:
        parser.error("Missmatch between number of plates specified and the number of matrices specified.")

