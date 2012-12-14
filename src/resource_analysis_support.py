"""
Support functions for the analysis module such that its fundaments
can be slimmed... this may be a temporary solution (that is it may
be moved further in the future, but main idea is to compartmentalize
and modularize stuff here)
"""
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

import sys
import os
import time
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from PIL import Image


#
# SCANNOMATIC LIBRARIES
#

import resource_app_config

#
# GLOBALS
#

_logger = None


#
# FUNCTIONS
#


def set_logger(logger):

    global _logger
    _logger = logger


def get_active_plates(meta_data, suppress_analysis, graph_watch):
    """Makes list of only relevant plates according to how
    analysis was started"""

    config = resource_app_config.Config()

    plate_position_keys = []

    if meta_data['Version'] >= config.version_first_pass_change_1:
        v_offset = 1
    else:
        v_offset = 0

    for i in xrange(len(meta_data['Pinning Matrices'])):

        if (suppress_analysis != True or graph_watch[0] == i) and\
            meta_data['Pinning Matrices'][i] is not None:

            plate_position_keys.append("plate_{0}_area".format(i + v_offset))

    plates = len(plate_position_keys)

    return plates, plate_position_keys


def verify_outdata_directory(outdata_directory):
    """Verifies that outdata directory exists and if not tries to create
    one. Also corrects path so that it doesn't end with separator"""

    if not os.path.isdir(outdata_directory):
        dir_OK = False
        if not os.path.exists(outdata_directory):
            try:
                os.makedirs(outdata_directory)
                dir_OK = True
            except:
                pass
        if not dir_OK:
            logger.critical("ANALYSIS, Could not construct outdata directory,"
                + " could be a conflict")
            sys.exit()

    if outdata_directory[-1] == os.sep:
        return outdata_directory[:-1]
    else:
        return outdata_directory


def get_pinning_matrices(query, sep=':'):
    """The function takes a string and parses it
    for known pinning matrix formats"""

    PINNING_MATRICES = {(8, 12): ['8,12', '96'],
                        (16, 24): ['16,24', '384'],
                        (32, 48): ['32,48', '1536'],
                        (64, 96): ['64,96', '6144'],
                        None: ['none', 'no', 'n', 'empty', '-', '--']}

    plate_strings = query.split(sep)
    plates = len(plate_strings) * [None]

    for i, p in enumerate(plate_strings):

        result = [k for k, v in PINNING_MATRICES.items() \
                if p.lower().replace(" ", "").strip("()") in v]

        if len(result) == 1:

            plates[i] = result[0]

        elif len(result) > 1:

            logger.warning("Ambigous plate pinning matrix statement" + \
                    " '{0}'".format(p))
        else:

            logger.warning(
                "Bad pinning pattern '{0}' - ignoring that plate".format(p))

    return plates


def print_progress_bar(fraction=0.0, size=40, start_time=None):
    """Prints an ascii progress bar"""
    prog_str = "["
    percent = 100 * fraction
    pfraction = fraction * size
    pfraction = int(round(pfraction))

    prog_str = "[" + pfraction * "=" + (size - pfraction) * " " + "]"
    perc_str ="%.1f" % (percent) + " %"

    prog_l = len(prog_str)
    perc_l = len(perc_str)

    prog_str = prog_str[:prog_l/2 - perc_l/2] + perc_str + \
                prog_str[prog_l/2 + perc_l:]

    print "\r{0}".format(prog_str),

    if start_time is not None:

        elapsed = time.time() - start_time
        eta = elapsed / fraction + start_time

        print " ETA: {0}".format(time.asctime(time.localtime(eta))),

    sys.stdout.flush()


def custom_traceback(excType, excValue, traceback):
    """Custom traceback function"""

    global _logger

    run_file_path = "(sorry couldn't find the name," + \
            " but it is the analysis.run of your project)"

    if _logger is not None:

        for handler in _logger.handlers:

            try:

                run_file_path = handler.baseFilename

            except:

                pass

    _logger.critical("Uncaught exception -- An error in the code was" + \
        " encountered.\n" + \
        "The analysis needs to be re-run when the problem is fixed.\n" + \
        "If you are lucky, the problem may be solved by recompiling" + \
        " a new .analysis file for " + \
        "the project.\nIn any a way, please send " + \
        "the file {0} to martin.zackrisson@gu.se".format(run_file_path),
        exc_info=(excType, excValue, traceback))

    sys.exit(1)


def get_finds_fixture(name, directory=None):

    return True


def get_run_will_do_something(suppress_analysis, graph_watch, 
                meta_data, logger):


    #Verifying sanity of request: Suppression requires watching?
    if suppress_analysis:

        if graph_watch is None or len(graph_watch) == 0:

            logger.critical("ANALYSIS: You are effectively requesting to" +
                " do nothing,\nso I guess I'm done...\n(If you suppress" +
                " analysis of non-watched colonies, then you need to watch" +
                " one as well!)")
    
            return False

        elif graph_watch[0] >= len(meta_data['Pinning Matrices']) or \
                graph_watch[0] < 0 or \
                meta_data['Pinning Matrices'][graph_watch[0]] is None:

            logger.critical("ANALYSIS: That plate ({0}) does not exist"\
                .format(graph_watch[0]) + " or doesn't have a pinning!")

            return False

        else:

            pm = meta_data['Pinning Matrices'][graph_watch[0]]

            if graph_watch[1] >= pm[0] or graph_watch[1] < 0 or \
                    graph_watch[2] >= pm[1] or graph_watch[2] < 0:

                logger.critical("ANALYSIS: The watch colony cordinate" + \
                    " ({0}) is out of bounds on plate {1}.".format(
                    graph_watch[1:], graph_watch[0]))

                return False 

        return True

#
# CLASSES
#

class Watch_Graph(object):
    """The Watch Graph is a composite data graph for a colony"""

    PLATE = 0
    X = 1
    Y = 1

    def __init__(self, watch_id, outdata_directory):

        self._watch = watch_id
        self._path =  os.sep.join((
            outdata_directory,
            "watch_image__plate_{0}_pos_{1}_{2}.png".format(
                self._watch[self.PLATE],
                self._watch[self.X],
                self._watch[self.Y])))

        self._reading = []
        self._x_labels = []
        self._y_labels = []

        self._pict_target_width = 40

        #Font
        fontP = FontProperties()
        fontP.set_size('xx-small')

        self._figure = plt.figure()

        self._figure.subplots_adjust(hspace=2, wspace=2)

        #IMAGES AX
        self._im_ax = self._figure.add_subplot(411)
        self._im_ax.axis("off")
        self._im_ax.axis((0, (self._pict_target_width + 1) * 217, 0,
            self._pict_target_width * 3), frameon=False,
            title='Plate: {0}, position: ({1}, {2})'.format(
                self._watch[self.PLATE], self._watch[self.X],
                self._watch[self.Y]))


    def add_image(self):
        
        pass
        """
        x_labels.append(image_pos)
        pict_size = project_image.watch_grid_size
        pict_scale = pict_target_width / float(pict_size[1])

        if pict_scale < 1:

            pict_resize = (int(pict_size[0] * pict_scale),
                            int(pict_size[1] * pict_scale))

            plt_watch_1.imshow(Image.fromstring('L',
                    (project_image.watch_scaled.shape[1],
                    project_image.watch_scaled.shape[0]),
                    project_image.watch_scaled.tostring())\
                    .resize(pict_resize, Image.BICUBIC),
                    extent=(image_pos * pict_target_width,
                    (image_pos + 1) * pict_target_width - 1,
                    10, 10 + pict_resize[1]))

        tmp_results = []

        if project_image.watch_results is not None:

            for cell_item in project_image.watch_results.keys():

                for measure in project_image.watch_results[\
                                            cell_item].keys():

                    if type(project_image.watch_results[\
                                    cell_item][measure])\
                                    == np.ndarray or \
                                    project_image.watch_results[\
                                    cell_item][measure] is None:

                        tmp_results.append(np.nan)

                    else:

                        tmp_results.append(
                                project_image.watch_results[\
                                cell_item][measure])

                    if len(watch_reading) == 0:

                        plot_labels.append(cell_item + ':' + measure)

        watch_reading.append(tmp_results)

        """

    def finalize(self):

        pass
        """
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
        X = (np.arange(len(image_dictionaries), 0, -1) + 0.5) *\
                        pict_target_width

        for xlabel_pos in xrange(len(x_labels)):

            if xlabel_pos % 5 > 0:
                x_labels[xlabel_pos] = ""

        cur_plt_graph = ""
        plt_graph_i = 1

        for i in [x for x in range(len(gws)) if x not in omits]:

            ii = i + sum(map(lambda x: x < i, omits))

            Y_good_positions = np.where(np.isnan(Y[i, :]) == False)[0]

            if Y_good_positions.size > 0:

                try:

                    if Y[i, Y_good_positions].max() == \
                                Y[i, Y_good_positions].min():

                        scale_factor = 0

                    else:

                        scale_factor = 100 / \
                                float(Y[i, Y_good_positions].max() - \
                                Y[i, Y_good_positions].min())

                    sub_term = float(Y[i, Y_good_positions].min())

                    if plot_labels[ii] == "cell:area":

                        c_area = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "background:mean":

                        bg_mean = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "cell:pixelsum":

                        c_pixelsum = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "blob:pixelsum":

                        b_pixelsum = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "blob:area":

                        b_area = Y[i, Y_good_positions]

                    logger.debug("WATCH GRAPH:\n%s\n%s\n%s" % \
                                (str(plot_labels[ii]), str(sub_term),
                                str(scale_factor)))

                    logger.debug("WATCH GRAPH, Max %.2f Min %.2f." % \
                            (float(Y[i, Y_good_positions].max()),
                            float(Y[i, Y_good_positions].min())))

                    if cur_plt_graph != plot_labels[ii].split(":")[0]:

                        cur_plt_graph = plot_labels[ii].split(":")[0]

                        if plt_graph_i > 1:

                            plt_watch_curves.legend(loc=1, ncol=5, prop=fontP,
                                    bbox_to_anchor=(1.0, -0.45))

                        plt_graph_i += 1

                        plt_watch_curves = plt_watch_colony.add_subplot(4, 1,
                                    plt_graph_i, title=cur_plt_graph)

                        plt_watch_curves.set_xticks(X)

                        plt_watch_curves.set_xticklabels(x_labels,
                                    fontsize="xx-small", rotation=90)

                    if scale_factor != 0:

                        plt_watch_curves.plot(X[Y_good_positions],
                            (Y[i, Y_good_positions] - sub_term) * scale_factor,
                            label=plot_labels[ii][len(cur_plt_graph) + 1:])

                    else:

                        logger.debug("GRAPH WATCH, Got straight line %s, %s" %\
                                (str(plt_graph_i), str(i)))

                        plt_watch_curves.plot(X[Y_good_positions],
                                np.zeros(X[Y_good_positions].shape) + \
                                10 * (i - (plt_graph_i - 1) * 5),
                                label=plot_labels[ii][len(cur_plt_graph) + 1:])

                except TypeError:

                    logger.warning("GRAPH WATCH, Error processing {0}".format(
                                                            plot_labels[ii]))

            else:

                    logger.warning("GRAPH WATCH, Cann't plot {0}".format(
                            plot_labels[ii]) + "since has no good data.")

        plt_watch_curves.legend(loc=1, ncol=5, prop=fontP,
                            bbox_to_anchor=(1.0, -0.45))

        if graph_output != None:

            try:

                plt_watch_colony.savefig(graph_output, dpi=300)

            except:

                plt_watch_colony.show()

            #DEBUG START:PLOT
            plt_watch_colony = pyplot.figure()
            plt_watch_1 = plt_watch_colony.add_subplot(111)
            plt_watch_1.loglog(b_area, b_pixelsum)
            plt_watch_1.set_xlabel('Blob:Area')
            plt_watch_1.set_ylabel('Blob:PixelSum')
            plt_watch_1.set_title('')
            plt_watch_colony.savefig("debug_corr.png")
            #DEBUG END

            pyplot.close(plt_watch_colony)

        else:

            plt_watch_colony.show()
        """

