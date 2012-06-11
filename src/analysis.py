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
__version__ = "0.993"
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
from matplotlib import pyplot
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
import resource_fixture as r_fixture
import resource_log_edit as rle        

#
# GLOBALS
#


#
# FUNCTIONS
#

def get_pinning_matrices(query, sep=':'):


    PINNING_MATRICES = {(8,12): ['8,12','96'],
                        (16,24): ['16,24','384'],
                        (32,48): ['32,48','1536'],
                        (64,96): ['64,96','6144'],
                        None: ['none','no','n','empy','-','--']}

    plate_strings = query.split(sep)
    plates = len(plate_strings) * [None]

    for i,p in enumerate(plate_strings):

        result = [k for k,v in PINNING_MATRICES.items() if p.lower().replace(" ","").strip("()") in v]

        if len(result) == 1:
            plates[i] = result[0]

        elif len(result) > 1:
            logger.warning("Ambigous plate pinning matrix statement '{0}'".format(p))
        else:
            logger.warning("Bad pinning pattern '{0}' - ignoring that plate".format(p))

    return plates


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
        grid_times=None, xml_format={'short': True, 'omit_compartments':[],
        'omit_measures':[]}, animate=False):
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

        @xml_format         A dict for what should be in the xml and
                            which tags (long vs short) be used.

        @animate            Boolean def (False) to cause saving animation
                            images.

        The function returns None if nothing was done of if it crashed.
        If it runs through it returns the number of images processed.

    """
    
    start_time = time()

    graph_output = None
    file_path_base = os.sep.join(log_file_path.split(os.sep)[:-1])

    #XML STATIC TEMPLATES
    XML_OPEN = "<{0}>"
    XML_OPEN_W_ONE_PARAM = '<{0} {1}="{2}">'
    XML_OPEN_CONT_CLOSE = "<{0}>{1}</{0}>" 
    XML_OPEN_W_ONE_PARAM_CONT_CLOSE = '<{0} {1}="{2}">{3}</{0}>'
    XML_OPEN_W_TWO_PARAM = '<{0} {1}="{2}" {3}="{4}">'

    XML_CLOSE = "</{0}>"

    XML_CONT_CLOSE = "{0}</{1}>"

    XML_SINGLE_W_THREE_PARAM = '<{0} {1}="{2}" {3}="{4}" {5}="{6}" />'
    #END XML STATIC TEMPLATES

    if not os.path.isdir(outdata_files_path):
        dir_OK = False
        if not os.path.exists(outdata_files_path):
            try:
                os.makedirs(outdata_files_path)
                dir_OK = True
            except:
                pass
        if not dir_OK:
            logger.critical("ANALYSIS, Could not construct outdata directory,"\
                + " could be a conflict")
            sys.exit()

    if outdata_files_path[-1] != os.sep:
        outdata_files_path += os.sep

    
    

    hdlr = logging.FileHandler(outdata_files_path + "analysis.run", mode='w')
    hdlr.setFormatter(log_formatter)
    logger.addHandler(hdlr)
    logger.info('Analysis started at ' + str(start_time))

    if graph_watch != None:
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
        logger.critical("ANALYSIS: Log file seems corrupt - remake one!")
        return None

    fixture_name = 'fixture_a'
    p_uuid = None
    grid_adjustments = None

    if 'Description' not in image_dictionaries[0].keys():
        fake_proj_metadata = rle.create_place_holder_meta_info(path = log_file_path)
        rle.rewrite(log_file_path, {0: {'mode':'a', 
            'data': str(fake_proj_metadata) + '\n'}})

        log_file = conf.Config_File(log_file_path)

        image_dictionaries = log_file.get_all("%n")
        if image_dictionaries == None:
            logger.critical("ANALYSIS: Log file seems corrupt - remake one!")
            return None

    if 'Description' in image_dictionaries[0].keys():
        first_scan_position = 1
        description = image_dictionaries[0]['Description']
        interval_time = image_dictionaries[0]['Interval']
        if pinning_matrices is None and 'Pinning Matrices' \
            in image_dictionaries[0].keys():

            pinning_matrices = image_dictionaries[0]['Pinning Matrices']
        if 'Fixture' in image_dictionaries[0].keys():
            fixture_name = image_dictionaries[0]['Fixture']

        if 'UUID' in image_dictionaries[0].keys():
            p_uuid =  image_dictionaries[0]['UUID']
            
        if 'Grid Adjustments' in image_dictionaries[0].keys():
            grid_adjustments =  image_dictionaries[0]['Grid Adjustments']

    else:
        first_scan_position = 0
        description = None
        interval_time = None

    if pinning_matrices is None:

        logger.critical("ANALYSIS: need some pinning matrices to analyse anything")
        return False

    plate_position_keys = []

    for i in xrange(len(pinning_matrices)):
        if (supress_analysis != True or graph_watch[0] == i) and\
            pinning_matrices[i] is not None:

            plate_position_keys.append("plate_" + str(i) + "_area")

    plates = len(plate_position_keys)
 
    if supress_analysis == True:
        project_image = Project_Image([pinning_matrices[graph_watch[0]]], 
            animate=animate, file_path_base=file_path_base, fixture_name=fixture_name,
            p_uuid=p_uuid)
        graph_watch[0] = 0
        plates = 1
    else:
        outdata_analysis_path = outdata_files_path + "analysis.xml"
        outdata_analysis_slimmed_path = outdata_files_path + "analysis_slimmed.xml"

        try:
            fh = open(outdata_analysis_path,'w')
            fhs = open(outdata_analysis_slimmed_path, 'w')

        except:
            logger.critical("ANALYSIS: can't open target file:'%s' or '%s'" % \
                (str(outdata_analysis_path),
                str(outdata_analysis_slimmed_path)))
            return False
        project_image = Project_Image(pinning_matrices, animate=animate,
            file_path_base = file_path_base, fixture_name=fixture_name,
            p_uuid = p_uuid)

    if grid_adjustments is not None:
        
        logger.info("ANALYSIS: Will implement manual adjustments of grid on plates {0}".format(\
            grid_adjustments.keys()))

        project_image.set_manual_grids(grid_adjustments)

    image_pos = len(image_dictionaries) - 1

    logger.info("ANALYSIS: A total of {0} images to analyse in project with UUID {1}".format(\
        len(image_dictionaries)-first_scan_position, p_uuid))

    if image_pos < first_scan_position:
        logger.critical("ANALYSIS: There are no images to analyse, aborting")
        for f in (fh, fhs):
            f.close()
        return True

    image_tot = image_pos 


    if supress_analysis != True:

        d_type_dict = {('pixelsum','ps'):('cells','standard'),
            ('area','a'):('pixels','standard'),
            ('mean','m'):('cells/pixel','standard'),
            ('median','md'):('cells/pixel','standard'),
            ('centroid','cent'):('(pixels,pixels)','coordnate'),
            ('perimeter','per'):('((pixels, pixels) ...)','list of coordinates'),
            ('IQR','IQR'):('cells/pixel to cells/pixel', 'list of standard'),
            ('IQR_mean','IQR_m'):('cells/pixel', 'standard')}


        for f in (fh, fhs):

            f.write('<project>')

            f.write(XML_OPEN_CONT_CLOSE.format(['start-time','start-t'][xml_format['short']],
                str(image_dictionaries[first_scan_position]['Time'])  ))

            f.write(XML_OPEN_CONT_CLOSE.format(['description','desc'][xml_format['short']],
                str(description) ))

            f.write(XML_OPEN_CONT_CLOSE.format(['number-of-scans','n-scans'][xml_format['short']],
                str(image_pos+1)))

            f.write(XML_OPEN_CONT_CLOSE.format(['interval-time','int-t'][xml_format['short']],
                str(interval_time)))

            f.write(XML_OPEN_CONT_CLOSE.format(['plates-per-scan','n-plates'][xml_format['short']],
                str(plates) ))

            f.write(XML_OPEN.format(['pinning-matrices','matrices'][xml_format['short']]))

            p_string = ""

            for pos in xrange(len(pinning_matrices)):
                if pinning_matrices[pos] is not None:
                    f.write(XML_OPEN_W_ONE_PARAM_CONT_CLOSE.format(\
                        ['pinning-matrix','p-m'][xml_format['short']],
                        ['index','i'][xml_format['short']],  str(pos), 
                        str(pinning_matrices[pos])))
                    p_string += "Plate {0}: {1}\t".format(pos, pinning_matrices[pos])
            logger.debug(p_string)

            f.write(XML_CLOSE.format(['pinning-matrices','matrices'][xml_format['short']]))

            f.write(XML_OPEN.format('d-types'))

            for d_type, info in d_type_dict.items():

                f.write(XML_SINGLE_W_THREE_PARAM.format(\
                    'd-type',
                    ['measure','m'][xml_format['short']],
                    d_type[xml_format['short']],
                    ['unit','u'][xml_format['short']],
                    info[0],
                    ['type','t'][xml_format['short']],
                    info[1]))

            f.write(XML_CLOSE.format('d-types'))

            f.write(XML_OPEN.format('scans'))

    logger.info("Starting analysis of {0} images, with log-record {1} (first image at {2})".\
        format(image_pos - first_scan_position + 1, image_pos, first_scan_position))

    
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

        logger.info("ANALYSIS, Running analysis on '{}'".format( \
            img_dict_pointer['File']))

        features = project_image.get_analysis( img_dict_pointer['File'], \
            plate_positions, img_dict_pointer['grayscale_values'], \
            use_fallback, use_otsu, watch_colony=graph_watch, \
            supress_other=supress_analysis, \
            save_graph_image=(image_pos in grid_times), \
            save_graph_name=outdata_files_path+"time_" + str(image_pos).zfill(4) +\
             "_plate_", grid_lock = True, identifier_time=image_pos, 
            timestamp=img_dict_pointer['Time'])

        if supress_analysis != True:

            for f in (fh, fhs):
                f.write(XML_OPEN_W_ONE_PARAM.format(['scan', 's'][xml_format['short']],  
                    ['index','i'][xml_format['short']], str(image_pos) ))
                f.write(XML_OPEN.format(['scan-valid','ok'][xml_format['short']]))

        if features is None:
            if supress_analysis != True:
                for f in (fh, fhs):
                    f.write(XML_CONT_CLOSE.format(0, \
                        ['scan-valid','ok'][xml_format['short']]))
                    f.write(XML_CLOSE.format(['scan', 's'][xml_format['short']]))
        else:
            if graph_watch != None and  project_image.watch_grid_size != None:

                x_labels.append(image_pos)
                pict_size = project_image.watch_grid_size
                pict_scale = pict_target_width / float(pict_size[1])
                if pict_scale < 1:
                    pict_resize = (int(pict_size[0] * pict_scale), int(pict_size[1] * pict_scale))
                    

                    plt_watch_1.imshow(Image.fromstring('L', (project_image.watch_scaled.shape[1], \
                        project_image.watch_scaled.shape[0]), \
                        project_image.watch_scaled.tostring()).resize(pict_resize, Image.BICUBIC), \
                        extent=(image_pos * pict_target_width, (image_pos+1)*pict_target_width-1, 10, 10 + pict_resize[1]))


                    #plt_watch_1.imshow(Image.fromstring('L', (project_image.watch_blob.shape[1], \
                        #project_image.watch_blob.shape[0]), \
                        #project_image.watch_blob.tostring()).resize(pict_resize, Image.BICUBIC), \
                        #extent=(image_pos * (pict_target_width), (image_pos+1)*(pict_target_width)-1, 10 + pict_resize[1] + 1, 10 + 2 * pict_resize[1] ))



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

                for f in (fh, fhs):
                    f.write(XML_CONT_CLOSE.format(1, 
                        ['scan-valid>','ok'][xml_format['short']]))

                    f.write(XML_OPEN_CONT_CLOSE.format(\
                        ['calibration','cal'][xml_format['short']],
                        str(img_dict_pointer['grayscale_values']) ))

                    f.write(XML_OPEN_CONT_CLOSE.format(\
                        ['time','t'][xml_format['short']],
                        str(img_dict_pointer['Time']) ))

                    f.write(XML_OPEN.format(['plates','pls'][xml_format['short']]))

                for i in xrange(plates):
                    for f in (fh, fhs):
                        f.write(XML_OPEN_W_ONE_PARAM.format(\
                            ['plate','p'][xml_format['short']],
                            ['index', 'i'][xml_format['short']], str(i) ))

                        f.write(XML_OPEN_CONT_CLOSE.format(\
                            ['plate-matrix','pm'][xml_format['short']],
                            str(pinning_matrices[i]) ))

                        f.write(XML_OPEN_CONT_CLOSE.format('R',
                            str(project_image.R[i])))

                        f.write(XML_OPEN.format(\
                            ['grid-cells','gcs'][xml_format['short']]))

                    for x, rows in enumerate(features[i]):
                        for y, cell in enumerate(rows):

                            for f in (fh, fhs):
                                f.write(XML_OPEN_W_TWO_PARAM.format(\
                                    ['grid-cell','gc'][xml_format['short']],
                                    'x', str(x), 
                                    'y', str(y)))

                            if cell != None:
                                for item in cell.keys():


                                    i_string = item

                                    if xml_format['short']:

                                        i_string = i_string\
                                                .replace('background','bg')\
                                                .replace('blob','bl')\
                                                .replace('cell','cl')

                                    if item not in xml_format['omit_compartments']:
                                        fhs.write(XML_OPEN.format(i_string))

                                    fh.write(XML_OPEN.format(i_string))
                                    
                                    for measure in cell[item].keys():

                                        m_string = XML_OPEN_CONT_CLOSE.format(\
                                            str(measure),
                                            str(cell[item][measure]))

                                        if xml_format['short']:
            
                                            m_string = m_string\
                                                    .replace('area','a')\
                                                    .replace('pixel','p')\
                                                    .replace('mean','m')\
                                                    .replace('median','md')\
                                                    .replace('sum','s')\
                                                    .replace('centroid','cent')\
                                                    .replace('perimeter','per')

                                        if item not in xml_format['omit_compartments'] and\
                                            measure not in xml_format['omit_measures']:

                                            fhs.write(m_string)
                                        fh.write(m_string)

                                    if item not in xml_format['omit_compartments']:
                                        fhs.write(XML_CLOSE.format(i_string))
                                    fh.write(XML_CLOSE.format(i_string))

                            for f in (fh, fhs):
                                f.write(XML_CLOSE.format(\
                                    ['grid-cell','gc'][xml_format['short']]))
                    for f in (fh, fhs):
                        f.write(XML_CLOSE.format(\
                            ['grid-cells','gcs'][xml_format['short']]))
                        f.write(XML_CLOSE.format(\
                            ['plate','p'][xml_format['short']]))
                for f in (fh, fhs):
                    f.write(XML_CLOSE.format(\
                        ['plates','pls'][xml_format['short']]))
                    f.write(XML_CLOSE.format(\
                        ['scan', 's'][xml_format['short']]))
        image_pos -= 1

        #DEBUGHACK
        #if image_pos > 1:
        #    image_pos = 1 
        #DEBUGHACK - END


        logger.info("ANALYIS, Image took %.2f seconds" % (time() - scan_start_time))

        print_progress_bar((image_tot-image_pos)/float(image_tot), size=70)

    if supress_analysis != True:
        for f in (fh, fhs):
            f.write(XML_CLOSE.format('scans'))
            f.write(XML_CLOSE.format('project'))
            f.close()

    if  graph_watch != None:
        #        print watch_reading

        omits = []
        gws = []
        for i in xrange(len(watch_reading[0])):
            gw_i = [gw[i] for gw in watch_reading]
            try:
                map(lambda v: len(v), gw_i)
                omits.append(i)
                gw_i = None
            except:
                pass

            if gw_i is not None:
                gws.append(gw_i)

        Y = np.asarray(gws, dtype=np.float64)
        X = (np.arange(len(image_dictionaries),0,-1)+0.5)*pict_target_width

        for xlabel_pos in xrange(len(x_labels)):
            if xlabel_pos % 5 > 0:
                x_labels[xlabel_pos] = ""
        #print len(X), len(x_labels)
        #print X
        #print x_labels
        cur_plt_graph = ""
        plt_graph_i = 1
        for i in [x for x in range(len(gws)) if x not in omits]:
            ii = i + sum(map(lambda x: x<i, omits))
            Y_good_positions = np.where(np.isnan(Y[i,:]) == False)[0]
            if Y_good_positions.size > 0:
                try:

        

                    if Y[i,Y_good_positions].max() == Y[i,Y_good_positions].min():
                        scale_factor = 0
                    else:
                        scale_factor =  100 / float(Y[i,Y_good_positions].max() - \
                            Y[i,Y_good_positions].min())

                    sub_term = float(Y[i,Y_good_positions].min())

                    if plot_labels[ii] == "cell:area":
                        c_area = Y[i,Y_good_positions]
                    elif plot_labels[ii] == "background:mean":
                        bg_mean = Y[i,Y_good_positions]
                    elif plot_labels[ii] == "cell:pixelsum":
                        c_pixelsum = Y[i,Y_good_positions]
                    elif plot_labels[ii] == "blob:pixelsum":
                        b_pixelsum = Y[i,Y_good_positions]
                    elif plot_labels[ii] == "blob:area":
                        b_area = Y[i,Y_good_positions]

                    logger.debug("WATCH GRAPH:\n%s\n%s\n%s" % \
                        (str(plot_labels[ii]), str(sub_term), 
                        str(scale_factor)))

                    #print ", NaNs: ", Y[:,i].size - Y[Y_good_positions,i].size
                    logger.debug("WATCH GRAPH, Max %.2f Min %.2f." % \
                        (float(Y[i,Y_good_positions].max()), 
                        float(Y[i,Y_good_positions].min())))

                    #print Y[Y_good_positions,i]
                    if cur_plt_graph != plot_labels[ii].split(":")[0]:
                        cur_plt_graph = plot_labels[ii].split(":")[0]
                        if plt_graph_i > 1:
                            plt_watch_curves.legend(loc=1, ncol=5, prop=fontP, 
                                bbox_to_anchor = (1.0, -0.45))

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
                            (Y[i,Y_good_positions] - sub_term) * scale_factor,
                            label=plot_labels[ii][len(cur_plt_graph)+1:])                        
                    else:
                        logger.debug("GRAPH WATCH, Got straight line %s, %s" % \
                            (str(plt_graph_i), str(i)))

                        plt_watch_curves.plot(X[Y_good_positions], 
                            np.zeros(X[Y_good_positions].shape)+\
                            10*(i-(plt_graph_i-1)*5), 
                            label=plot_labels[ii][len(cur_plt_graph)+1:])


                except TypeError:
                    logger.warning("GRAPH WATCH, Error processing %s" % str(plot_labels[ii]))

            else:
                    logger.warning("GRAPH WATCH, Cann't plot %s since has \
no good data" % str(plot_labels[ii]))

        plt_watch_curves.legend(loc=1, ncol=5, prop=fontP, 
            bbox_to_anchor = (1.0, -0.45))
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

        logger.info("ANALYSIS, Full analysis took %.2f minutes" %\
            ((time() - start_time)/60.0))

        logger.info('Analysis completed at ' + str(time()))

        return False


    logger.info("ANALYSIS, Full analysis took %.2f minutes" %\
        ((time() - start_time)/60.0))

    logger.info('Analysis completed at ' + str(time()))
#
# CLASS Project_Image
#

class Project_Image():
    def __init__(self, pinning_matrices, im_path=None, plate_positions=None,
        animate=False, file_path_base="", fixture_name='fixture_a',
        p_uuid=None, logger = None):


        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger('Scan-o-Matic Analysis')

        self.p_uuid = p_uuid

        self._im_path = im_path
        self._im_loaded = False

        self._animate = animate
        
        self._plate_positions = plate_positions
        self._pinning_matrices = pinning_matrices

        #PATHS
        script_path_root = os.path.dirname(os.path.abspath(__file__))
        scannomatic_root = os.sep.join(script_path_root.split(os.sep)[:-1])
        self._program_root = scannomatic_root
        self._program_code_root = scannomatic_root + os.sep + "src"
        self._program_config_root = self._program_code_root + os.sep + "config"
        self._file_path_base = file_path_base

        self.fixture = r_fixture.Fixture_Settings(\
            self._program_config_root + os.sep + "fixtures", 
            fixture = fixture_name)

        self.im = None
        self._grid_arrays = []
        self.features = []
        self.R = []

        self._timestamp = None
        self._pinning_matrices = pinning_matrices

        for a in xrange(len(pinning_matrices)):
            if pinning_matrices[a] is not None:
                self._grid_arrays.append(grid_array.Grid_Array(self, (a,), pinning_matrices[a]))
                self.features.append(None)
                self.R.append(None)

        if len(pinning_matrices) > len(self._grid_arrays):
            logger.info('Analysis will run on {0} plates out of {1}'.format(\
                len(self._grid_arrays), len(pinning_matrices)))

    def set_manual_grids(self, grid_adjustments):
        """
            Overrides grid detection with a specified grid supplied in grid adjustments

            @param grid_adjustments:    A dictionary of pinning grids with plate numbers as
                                        keys and items being tuples of row and column position 
                                        lists.
        """

        for k in grid_adjustments.keys():

            if self._pinning_matrices[k] is not None:

                try:
                    self._grid_arrays[k].set_manual_grid(grid_adjustments[k])
                except IndexError:
                    logger.error('Failed to set manual grid adjustments to {0}, plate non-existent'.\
                        format(k))
            

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

            alt_path = os.sep.join((self._file_path_base,
                self._im_path.split(os.sep)[-1]))

            logger.warning("ANALYSIS IMAGE, Could not open image at '{0}' trying in log-file directory ('{1}').".\
                format(self._im_path, alt_path))

            #self._im_path = os.sep.join((self._file_path_base, 
            #    self.im_path.split(os.sep)[-1]))

            try:
                self.im = plt_img.imread(alt_path)
                self._im_loaded = True
            except:
                logger.warning("ANALYSIS IMAGE, No image found... sorry")
        
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

        logger.debug("ANALYSIS produced gs-coefficients {0} ".format(gs_fit))

        if gs_fit is not None:

            z3_deriv_coeffs = np.array(gs_fit[:-1]) * np.arange(gs_fit.shape[0]-1,0,-1)

            z3_deriv = np.array(map(lambda x: (z3_deriv_coeffs*np.power(x,
                np.arange(z3_deriv_coeffs.shape[0],0,-1))).sum(), range(87)))

            if (z3_deriv > 0).any() and (z3_deriv < 0).any():

                logger.warning("ANALYSIS of grayscale seems dubious as \
coefficients don't have the same sign")
                gs_fit = None


        if gs_fit is None:

            return None    

        scale_factor = 4.0

        if save_graph_image == False:
            cur_graph_name = None

        ###DEBUG GRID ARRAYS
        #print "The", len(self._grid_arrays), "arrays:", self._grid_arrays
        ###DEBUG END

        for grid_array in xrange(len(self._grid_arrays)):

            x0 = round(features[grid_array][0][0]*scale_factor)
            x1 = round(features[grid_array][1][0]*scale_factor)
            if x0 < x1:
                upper = x0
                lower = x1
            else:
                upper = x1
                lower = x0

            y0 = round(features[grid_array][0][1]*scale_factor)
            y1 = round(features[grid_array][1][1]*scale_factor)
            if y0 < y1:
                left = y0
                right = y1
            else:
                left = y1
                right = y0

 
            if save_graph_image == True:

                cur_graph_name = save_graph_name + str(grid_array) + ".png"
            save_anime_name = save_graph_name + "anime.png"
            
            self._grid_arrays[grid_array].get_analysis( \
                self.im[ upper:lower, left:right ], \
                gs_values=gs_values, use_fallback=use_fallback,\
                use_otsu=use_otsu, median_coeff=None, \
                verboise=False, visual=False, watch_colony=watch_colony, \
                supress_other=supress_other, save_grid_image=save_graph_image\
                , save_grid_name = cur_graph_name, 
                save_anime_name = save_anime_name, grid_lock = grid_lock,
                identifier_time = identifier_time, timestamp=timestamp,
                animate=self._animate)

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
    parser.add_argument("-m", "--matrices", dest="matrices", default=None, help="The pinning matrices for each plate position in the order set by the fixture config", metavar="(X,Y):(X,Y)...(X,Y)")

    parser.add_argument("-w", "--watch-position", dest="graph_watch", help="The position of a colony to track.", metavar="PLATE:X:Y", type=str)

    #parser.add_argument("-g", "--graph-output", dest="graph_output", help="If specified the graph is not shown to the user but instead saved to taget position", type=str)

    parser.add_argument("-t", "--watch-time", dest="grid_times", help="If specified, the gridplacements at the specified timepoints will be saved in the set output-directory, comma-separeted indices.", metavar="0,1,100", default="0", type=str)

    parser.add_argument('-a', '--animate', dest="animate", default=False, type=bool, help="If True, it will produce stop motion images of the watched colony ready for animation")

    parser.add_argument("-s", "--supress-analysis", dest="supress", default=False, type=bool, help="If submitted, main analysis will be by-passed and only the plate and position that was specified by the -w flag will be analysed and reported.")

    parser.add_argument("--xml-short", dest="xml_short", default=False, type=bool, 
        help="If the XML output should use short tag-names")

    parser.add_argument("--xml-omit-compartments", dest="xml_omit_compartments", type=str, 
        help="Comma separated list of compartments to not report")

    parser.add_argument("--xml-omit-measures", dest="xml_omit_measures", type=str,
        help="Comma seperated list of measures to not report")
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

    
    logger = logging.getLogger('Scan-o-Matic Analysis')

    log_formatter = logging.Formatter('\n\n%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S\n')
    
    #XML
    xml_format = {'short':args.xml_short,'omit_compartments':[], 'omit_measures': []}

    if args.xml_omit_compartments is not None:

        xml_format['omit_compartments'] = map(lambda x: x.strip(), 
            args.xml_omit_compartments.split(","))

    if args.xml_omit_measures is not None:

        xml_format['omit_measures'] = map(lambda x: x.strip(),
            args.xml_omit_measures.split(","))

    
    logging.debug("XML-formatting is {0}, omitting compartments {1} and measures {2}."\
        .format(['long','short'][xml_format['short']], 
        str(xml_format['omit_compartments']), str(xml_format['omit_measures'])))


    #MATRICES
    if args.matrices is not None:

        pm = get_pinning_matrices(args.matrices)
        logging.debug("Matrices: {0}".format(pm))


    else:

        pm = None


    if args.grid_times != None:

        try:
            grid_times = (int(grid_times),)
        except:
            try:
                grid_times = map(int, args.grid_times.split(","))
            except:
                logging.warning("ARGUMENTS, could not parse grid_times... will only save\
     the first grid placement.")

                grid_times = [-1]

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

    logger.setLevel(logging_level)
    logger.debug("Logger is ready!")
    analyse_project(args.inputfile, output_path, pm, args.graph_watch, 
        args.supress, True, False, False, grid_times=grid_times,
        xml_format = xml_format, animate=args.animate)
    #else:
        #parser.error("Missmatch between number of plates specified and the number of matrices specified.")

