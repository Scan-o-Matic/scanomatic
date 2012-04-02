#!/usr/bin/env python
"""
This module is the typical starting-point of the analysis work-flow.
It has command-line behaviour but can also be run as part of another program.
It should be noted that a full run of around 200 images takes more than 2h on
a good computer using 100% of one processor. That is, if run from within 
another application, it is probably best to run it as a subprocess.
"""
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

import os, sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.image as plt_img
import types
import logging
import numpy as np
from time import time
from argparse import ArgumentParser

#
# SCANNOMATIC LIBRARIES
#

import analysis_grid_array as grid_array
import resource_config as conf

#
# FUNCTIONS
#

def print_progress_bar(fraction, size=40):
    prog_str = "["
    percent = 100 * fraction
    fraction *= size
    fraction = int(round(fraction))

    prog_str = "[" + fraction*"=" + (size-fraction)*" " + "]"

    print "\r%s %.1f" % (prog_str, percent),
    sys.stdout.flush()

def analyse_project(log_file_path, outdata_files_path, pinning_matrices, \
        graph_watch, supress_analysis = False, \
        verboise=False, use_fallback = False, use_otsu = False,\
        grid_times=None):
    """
        analyse_project parses a log-file and runs a full analysis on all 
        images in it. It will step backwards in time, starting with the 
        last image.

        The function takes the following arguments:

        @log_file_path      The path to the log-file to be processed

        @outdata_files_path  The path to the file were the analysis will
                            be put

        @pinning_matrices   A list/tuple of (row, columns) pinning 
                            matrices used for each plate position 
                            respectively

        @graph_watch        A coordinate PLATE:COL:ROW for a colony to
                            watch particularly.

        VOID@graph_output   An optional PATH to where to save the graph
                            produced by graph_watch being set.

        @supress_analysis  Suppresses the main analysis and thus
                            only graph_watch thing is produced.

        @verboise           Will print some basic output of progress.

        @use_fallback       Determines if fallback colony detection
                            should be used.

        @use_otsu           Determines if Otsu-thresholding should be
                            used when looking finding the grid (not
                            default, and should not be used)

        @grid_times         Specifies the time-point indices at which
                            the grids will be saved in the output-dir.

        The function returns None if nothing was done of if it crashed.
        If it runs through it returns the number of images processed.

    """
    
    start_time = time()
    graph_output = None

    if not os.path.isdir(outdata_files_path):
        dir_OK = False
        if not os.path.exists(outdata_files_path):
            try:
                os.makedirs(outdata_files_path)
                dir_OK = True
            except:
                pass
        if not dir_OK:
            logging.critical("ANALYSIS, Could not construct outdata directory,"\
                + " could be a conflict")
            sys.exit()

    if outdata_files_path[-1] != os.sep:
        outdata_files_path += os.sep


    
    try:
        fs = open(outdata_files_path + "analysis.run", 'w')
        fs.write('Analysis started at ' + str(start_time) + '\n')
        fs.close()
    except:
        logging.warning("Could not produce an 'analysis.run' file.")


    if graph_watch != None:
        from matplotlib import pyplot
        from matplotlib.font_manager import FontProperties
        from PIL import Image
        watch_reading = [] 
        x_labels = []

        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt_watch_colony = pyplot.figure()
        plt_watch_colony.subplots_adjust(hspace=2, wspace=2)
        plt_watch_1 = plt_watch_colony.add_subplot(411)
        plt_watch_1.axis("off")
        pict_target_width = 40
        plt_watch_1.axis((0, (pict_target_width + 1) * 217, 0, \
            pict_target_width * 3), frameon=False,\
            title='Plate: ' + str(graph_watch[0]) + ', position: (' +\
             str(graph_watch[1]) + ', ' + str(graph_watch[2]) + ')')

        plot_labels = []

        graph_output = outdata_files_path + "plate_" + str(graph_watch[0]) + \
            "_" + str(graph_watch[1]) + '.' + str(graph_watch[2]) + ".png"


    log_file = conf.Config_File(log_file_path)

    image_dictionaries = log_file.get_all("%n")
    if image_dictionaries == None:
        return None


    if 'Description' in image_dictionaries[0].keys():
        first_scan_position = 1
        description = image_dictionaries[0]['Description']
        interval_time = image_dictionaries[0]['Interval']
        if pinning_matrices is None and 'Pinning Matrices' \
            in image_dictionaries[0].keys():

            pinning_matrices = image_dictionaries[0]['Pinning Matrices']

    else:
        first_scan_position = 0
        description = None
        interval_time = None

    if pinning_matrices is None:

        logging.critical("ANALYSIS,  need some pinning matrices to analyse anything")
        return False

    plate_position_keys = []

    for i in xrange(len(pinning_matrices)):
        if (supress_analysis != True or graph_watch[0] == i) and\
            pinning_matrices[i] is not None:

            plate_position_keys.append("plate_" + str(i) + "_area")

    plates = len(plate_position_keys)
 
    if supress_analysis == True:
        project_image = Project_Image([pinning_matrices[graph_watch[0]]])
        graph_watch[0] = 0
        plates = 1
    else:
        outdata_analysis_path = outdata_files_path + "analysis.xml"
        try:
            fh = open(outdata_analysis_path,'w')
        except:
            logging.critical("ANALYSIS, can't open target file:'%s'" % \
                str(outdata_analysis_path))
            return False
        project_image = Project_Image(pinning_matrices)

    image_pos = len(image_dictionaries) - 1

    if image_pos < first_scan_position:
        logging.critical("ANALYSIS, There are no images to analyse, aborting")
        fh.close()
        return True

    image_tot = image_pos 

    logging.info("ANALYIS, starting project with %d images" % (image_pos + 1))

    if supress_analysis != True:
        fh.write('<project>')

        fh.write('<start-time>' + str(image_dictionaries[first_scan_position]['Time']) + '</start-time>')

        fh.write('<description>' + str(description) + '</description>')

        fh.write('<number-of-scans>' + str(image_pos+1) + '</number-of-scans>')

        fh.write('<interval-time>' + str(interval_time) + '</interval-time>')

        fh.write('<plates-per-scan>' + str(plates) + '</plates-per-scan>')

        fh.write('<pinning-matrices>')

        for pos in xrange(plates):
            fh.write('<pinning-matrix index="' + str(pos) + '">' + \
                 str(pinning_matrices[pos]) + '</pinning-matrix>')

        fh.write('</pinning-matrices>')

        fh.write('<scans>')

    while image_pos >= first_scan_position:
        scan_start_time = time()
        img_dict_pointer = image_dictionaries[image_pos]

        plate_positions = []

        for i in xrange(plates):
            plate_positions.append( \
                img_dict_pointer[plate_position_keys[i]] )

            #if verboise:
            #    print "** Position", plate_position_keys[i], ":", \
            #img_dict_pointer[plate_position_keys[i]]

        logging.info("ANALYSIS, Running analysis on '%s'" % \
            str(img_dict_pointer['File']))

        features = project_image.get_analysis( img_dict_pointer['File'], \
            plate_positions, img_dict_pointer['grayscale_values'], \
            use_fallback, use_otsu, watch_colony=graph_watch, \
            supress_other=supress_analysis, \
            save_graph_image=(image_pos in grid_times), \
            save_graph_name=outdata_files_path+"time_" + str(image_pos) +\
             "_plate_", grid_lock = True, identifier_time=image_pos, 
            timestamp=img_dict_pointer['Time'])

        if supress_analysis != True:
            fh.write('<scan index="' + str(image_pos) + '">')
            fh.write('<scan-valid>')

        if features == None:
            if supress_analysis != True:
                fh.write(str(0) + '</scan-valid>')
                fh.write('</scan>')
        else:
            if graph_watch != None and  project_image.watch_grid_size != None:

                x_labels.append(image_pos)
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
                        if type(project_image.watch_results[cell_item][measure])\
                                == np.ndarray or \
                                project_image.watch_results[cell_item][measure] \
                                is None:

                            tmp_results.append(np.nan)
                        else:
                            tmp_results.append(project_image.watch_results[cell_item][measure])
                        if len(watch_reading) == 0:
                            plot_labels.append(cell_item + ':' + measure)
                watch_reading.append(tmp_results)    
                
                #HACK START: DEBUGGING
                #print "*** R:", project_image.R
                #if image_pos < 200:
                #    image_pos = -1 
                #HACK END

            if supress_analysis != True:
                fh.write(str(1) + '</scan-valid>')

                fh.write('<calibration>' + str(img_dict_pointer['grayscale_values']) + '</calibration>')

                fh.write('<time>' + str(img_dict_pointer['Time']) + '</time>')

                fh.write('<plates>')

                for i in xrange(plates):
                    fh.write('<plate index="' + str(i) + '">')

                    fh.write('<plate-matrix>' + str(pinning_matrices[i]) + '</plate-matrix>')

                    fh.write('<R>' + str(project_image.R[i]) + '</R>')

                    fh.write('<grid-cells>')

                    for x, rows in enumerate(features[i]):
                        for y, cell in enumerate(rows):

                            fh.write('<grid-cell x="' + str(x) + '" y="' + str(y) + '">')

                            if cell != None:
                                for item in cell.keys():

                                    fh.write('<' + str(item) + '>')
                                    
                                    for measure in cell[item].keys():

                                        fh.write('<' + str(measure) + '>' + \
                                            str(cell[item][measure]) + \
                                            '</' + str(measure) + '>')

                                    fh.write('</' + str(item) + '>')
                            fh.write('</grid-cell>')
                    fh.write('</grid-cells>')
                    fh.write('</plate>')
                fh.write('</plates>')
                fh.write('</scan>')
        image_pos -= 1

        #DEBUGHACK
        #if image_pos > 1:
        #    image_pos = 1 
        #DEBUGHACK - END


        logging.info("ANALYIS, Image took %.2f seconds" % (time() - scan_start_time))

        print_progress_bar((image_tot-image_pos)/float(image_tot), size=70)

    if supress_analysis != True:
        fh.write('</scans>')
        fh.write('</project>')
        fh.close()

    if  graph_watch != None:
        #print watch_reading
        Y = np.asarray(watch_reading, dtype=np.float64)
        X = (np.arange(len(image_dictionaries),0,-1)+0.5)*pict_target_width

        for xlabel_pos in xrange(len(x_labels)):
            if xlabel_pos % 5 > 0:
                x_labels[xlabel_pos] = ""
        #print len(X), len(x_labels)
        #print X
        #print x_labels
        cur_plt_graph = ""
        plt_graph_i = 1
        for i in xrange(int(Y.shape[1])):
            if type(Y[0,i]) != np.ndarray and type(Y[0,i]) != types.ListType and type(Y[0,i]) != types.TupleType:
                Y_good_positions = np.where(np.isnan(Y[:,i]) == False)[0]
                if Y_good_positions.size > 0:
                    try:

            

                        if Y[Y_good_positions,i].max() == Y[Y_good_positions,i].min():
                            scale_factor = 0
                        else:
                            scale_factor =  100 / float(Y[Y_good_positions,i].max() - Y[Y_good_positions,i].min())

                        sub_term = float(Y[Y_good_positions,i].min())

                        if plot_labels[i] == "cell:area":
                            c_area = Y[Y_good_positions,i]
                        elif plot_labels[i] == "background:mean":
                            bg_mean = Y[Y_good_positions,i]
                        elif plot_labels[i] == "cell:pixelsum":
                            c_pixelsum = Y[Y_good_positions,i]
                        elif plot_labels[i] == "blob:pixelsum":
                            b_pixelsum = Y[Y_good_positions,i]
                        elif plot_labels[i] == "blob:area":
                            b_area = Y[Y_good_positions,i]

                        logging.debug("WATCH GRAPH:\n%s\n%s\n%s" % \
                            (str(plot_labels[i]), str(sub_term), 
                            str(scale_factor)))

                        #print ", NaNs: ", Y[:,i].size - Y[Y_good_positions,i].size
                        logging.debug("WATCH GRAPH, Max %.2f Min %.2f." % \
                            (float(Y[Y_good_positions,i].max()), 
                            float(Y[Y_good_positions,i].min())))

                        #print Y[Y_good_positions,i]
                        if cur_plt_graph != plot_labels[i].split(":")[0]:
                            cur_plt_graph = plot_labels[i].split(":")[0]
                            if plt_graph_i > 1:
                                plt_watch_curves.legend(loc=1, ncol=5, prop=fontP, bbox_to_anchor = (1.0, -0.45))
                            plt_graph_i += 1
                            plt_watch_curves = plt_watch_colony.add_subplot(4,1,plt_graph_i,
                                title=cur_plt_graph)
                            plt_watch_curves.set_xticks(X)
                            plt_watch_curves.set_xticklabels(x_labels, fontsize="xx-small", rotation=90)

                        if scale_factor != 0: 
                            #print X[Y_good_positions].shape, Y[Y_good_positions,i].shape
                            #print X[Y_good_positions]
                            #print Y[Y_good_positions,i]
                           
                            plt_watch_curves.plot(X[Y_good_positions], 
                                (Y[Y_good_positions,i] - sub_term) * scale_factor,
                                label=plot_labels[i][len(cur_plt_graph)+1:])                        
                        else:
                            logging.debug("GRAPH WATCH, Got straight line %s, %s" % \
                                (str(plt_graph_i), str(i)))

                            plt_watch_curves.plot(X[Y_good_positions], 
                                np.zeros(X[Y_good_positions].shape)+\
                                10*(i-(plt_graph_i-1)*5), 
                                label=plot_labels[i][len(cur_plt_graph)+1:])


                    except TypeError:
                        logging.warning("GRAPH WATCH, Error processing %s" % str(plot_labels[i]))

                else:
                        logging.warning("GRAPH WATCH, Cann't plot %s since has \
no good data" % str(plot_labels[i]))

        plt_watch_curves.legend(loc=1, ncol=5, prop=fontP, bbox_to_anchor = (1.0, -0.45))
        if graph_output != None:
            try:
                plt_watch_colony.savefig(graph_output, dpi=300)
            except:
                plt_watch_colony.show()

            #DEBUG START:PLOT
            plt_watch_colony = pyplot.figure()
            plt_watch_1 = plt_watch_colony.add_subplot(111)
            plt_watch_1.plot(b_area, b_pixelsum)
            plt_watch_1.set_xlabel('Blob:Area')
            plt_watch_1.set_ylabel('Blob:PixelSum')
            plt_watch_1.set_title('')
            plt_watch_colony.savefig("debug_corr.png")
            #DEBUG END
            pyplot.close(plt_watch_colony)
        else: 
            plt_watch_colony.show()

        logging.info("ANALYSIS, Full analysis took %.2f minutes" %\
            ((time() - start_time)/60.0))

        try:
            fs = open(outdata_files_path + "analysis.run", 'a')
            fs.write('Analysis completed at ' + str(time()) + '\n')
            fs.close()
        except:
            logging.warning("ANALYSIS: Could not add to 'analysis.run' file.")

        return False

    try:
        fs = open(outdata_files_path + "analysis.run", 'a')
        fs.write('Analysis completed at ' + str(time()) + '\n')
        fs.close()
    except:
        logging.warning("ANALYSIS: Could not add to 'analysis.run' file.")

    logging.info("ANALYSIS, Full analysis took %.2f minutes" %\
        ((time() - start_time)/60.0))

#
# CLASS Project_Image
#

class Project_Image():
    def __init__(self, pinning_matrices, im_path=None, plate_positions=None ):

        self._im_path = im_path
        self._im_loaded = False
        
        self._plate_positions = plate_positions
        self._pinning_matrices = pinning_matrices

        #PATHS
        script_path_root = os.path.dirname(os.path.abspath(__file__))
        scannomatic_root = os.sep.join(script_path_root.split(os.sep)[:-1])
        self._program_root = scannomatic_root
        self._program_code_root = scannomatic_root + os.sep + "src"
        self._program_config_root = self._program_code_root + os.sep + "config"

        self.im = None
        self._grid_arrays = []
        self.features = []
        self.R = []

        self._timestamp = None

        for a in xrange(len(pinning_matrices)):
            if pinning_matrices[a] is not None:
                self._grid_arrays.append(grid_array.Grid_Array(self, (a,), pinning_matrices[a]))
                self.features.append(None)
                self.R.append(None)

    def get_analysis(self, im_path, features, grayscale_values, \
            use_fallback=False, use_otsu=True, watch_colony=None, \
            supress_other=False, save_graph_image=False, save_graph_name=None,
            grid_lock=False, identifier_time = None, timestamp = None):

        """
            @param im_path: An path to an image

            @param features: A list of pinning grids to look for

            @param grayscale_values : An array of the grayscale pixelvalues, if 
            submittet gs_fit is disregarded

            @param use_fallback : Causes fallback detection to be used.

            @param use_otsu : Causes thresholding to be done by Otsu
            algorithm (Default)

            @param watch_colony : A particular colony to gather information
            about.

            @param supress_other : If only the watched colony should be 
            analysed

            @param save_grid_image : Causes the script to save the plates' 
            grid placement as images. Conflicts with visual, so don't use 
            visual if you want to save

            @param save_grid_name : A custom name for the saved image, if none
            is submitted, it will be grid.png in current directory.

            @param grid_lock : Default False, if true, the grid will only be
            gotten once and then reused all way through.

            @param identifier_time : A time index to update the identifier with

            The function returns two arrays, one per dimension, of the
            positions of the spikes and a quality index

        """

        if im_path != None:
            self._im_path = im_path

        try:
            self.im = plt_img.imread(self._im_path)
            self._im_loaded = True
        except:
            logging.warning("ANALYSIS IMAGE, Could not open image at '%s'" %\
                str(self._im_path))

            self._im_loaded = False


        if self._im_loaded == True:           
            if len(self.im.shape) > 2:
                self.im = self.im[:,:,0]
        else:
            return None

        self._timestamp = timestamp

        if len(grayscale_values) > 3:
            gs_values = np.array(grayscale_values)
            gs_indices = np.arange(len(grayscale_values))

            gs_fit = np.polyfit(gs_indices, gs_values,3)
        else:
            gs_fit = None

        scale_factor = 4.0

        if save_graph_image == False:
            cur_graph_name = None

        ###DEBUG GRID ARRAYS
        #print "The", len(self._grid_arrays), "arrays:", self._grid_arrays
        ###DEBUG END

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

 
            if save_graph_image == True:

                cur_graph_name = save_graph_name + str(grid_array) + ".png"
            
            self._grid_arrays[grid_array].get_analysis( \
                self.im[ upper:lower, left:right ], \
                gs_values=gs_values, use_fallback=use_fallback,\
                use_otsu=use_otsu, median_coeff=None, \
                verboise=False, visual=False, watch_colony=watch_colony, \
                supress_other=supress_other, save_grid_image=save_graph_image\
                , save_grid_name = cur_graph_name, grid_lock = grid_lock,
                identifier_time = identifier_time, timestamp=timestamp)

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
    parser.add_argument("-o", "--ouput-path", type=str, dest="outputpath", help="Path to directory where all data is written (Default is a subdirectory 'analysis' under where the input file is)", metavar="PATH")

    #parser.add_argument("-p", "--plates", default=4, type=int, dest="plates", help="The number of plates in the fixture", metavar="N")
    parser.add_argument("-m", "--matrices", dest="matrices", help="The pinning matrices for each plate position in the order set by the fixture config", metavar="(X,Y):(X,Y)...(X,Y)")

    parser.add_argument("-w", "--watch-position", dest="graph_watch", help="The position of a colony to track.", metavar="PLATE:X:Y", type=str)

    #parser.add_argument("-g", "--graph-output", dest="graph_output", help="If specified the graph is not shown to the user but instead saved to taget position", type=str)

    parser.add_argument("-t", "--watch-time", dest="grid_times", help="If specified, the gridplacements at the specified timepoints will be saved in the set output-directory, comma-separeted indices.", metavar="0,1,100", default="0", type=str)

    parser.add_argument("-s", "--supress-analysis", dest="supress", default=False, type=bool, help="If set to True, main analysis will be by-passed and only the plate and position that was specified by the -w flag will be analysed and reported.")

    parser.add_argument("--debug", dest="debug_level", default="warning", type=str, help="Set debugging level")    

    args = parser.parse_args()

    #DEBUGGING
    LOGGING_LEVELS = {'critical': logging.CRITICAL,

                      'error': logging.ERROR,

                      'warning': logging.WARNING,

                      'info': logging.INFO,

                      'debug': logging.DEBUG}

    if args.debug_level in LOGGING_LEVELS.keys():

        logging_level = LOGGING_LEVELS[args.debug_level]

    else:

        logging_level = LOGGING_LEVELS['warning']

    logging.basicConfig(level=logging_level, format='\n\n%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S\n')

    #MATRICES
    if args.matrices is not None:

        pm = args.matrices.split(':')
        pm = map(eval, pm)
    else:

        pm = None

    if args.grid_times != None:

        try:
            grid_times = map(int, args.grid_times.split(","))
        except:
            logging.warning("ARGUMENTS, could not parse grid_times... will only save\
 the first grid placement.")

            grid_times = [0]

 
    if args.inputfile == None:
        parser.error("You need to specify input file!")

    in_path_list = args.inputfile.split(os.sep)

    output_path = ""

    if len(in_path_list) == 1:

        output_path = "."

    else:

        output_path = os.sep.join(in_path_list[:-1]) 
    if args.outputpath == None:


        output_path += os.sep + "analysis"
    else:
        output_path += os.sep + str(args.outputpath)
         
    if args.graph_watch != None:
        args.graph_watch = args.graph_watch.split(":")
        try:
            args.graph_watch = map(int, args.graph_watch)
        except:
            parser.error('The watched colony could not be resolved, make sure that you follow syntax')

        if len(args.graph_watch) <> 3:
            parser.error('Bad specification of watched colony')

        #if args.graph_watch[0] < args.plates:
            #if not(0 <= args.graph_watch[1] <= pm[args.graph_watch[0]][0] and 0 <= args.graph_watch[2] <= pm[args.graph_watch[0]][1]):
                #parser.error('The watched colony position is out of bounds (range: (0, 0)) - ' + str(pm[args.graph_watch[1]]) + ').')
        #else:
            #parser.error('The watched colony position has a plate number that is too high (max: ' + str(args.plates-1) + ').')

    try:
        fh = open(args.inputfile,'r')
    except:
        parser.error('Cannot open input file, please check your path...')

    fh.close()

    #if args.outputfile != None:
        #try:
            #fh = open(args.outputfile, 'w')
        #except:
            #parser.error('Cannot create the output file')
        #fh.close()


    #if args.graph_output != None:
        #try:
            #fh = open(args.graph_output, 'w')
        #except:
            #parser.error('Cannot create the save-file for the watched colony.')

        #fh.close()

    #if len(pm) == args.plates:    
    header_str = "The Project Analysis Script..."
    under_line = "-"
    print "\n\n%s\n%s\n\n" % (header_str.center(80), (len(header_str)*under_line).center(80))

    analyse_project(args.inputfile, output_path, pm, args.graph_watch, args.supress, True, False, False, grid_times=grid_times)
    #else:
        #parser.error("Missmatch between number of plates specified and the number of matrices specified.")

