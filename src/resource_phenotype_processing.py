#!/usr/bin/env python
#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.pylab as plt_lab
import logging
import scipy.interpolate as scint
#import scipy.stats as stats
import sys
import os
import pickle
import itertools

import resource_xml_read as r_xml


def make_linear_interpolation(data):
    y, x = np.where(data != 0)
    Y, X = np.arange(data.shape[0]), np.arange(data.shape[1])
    logging.warning("ready for linear")
    f = scint.interp2d(x, y, data[y, x], 'linear')
    logging.warning("done linear interpol")
    return f(X, Y)


def make_linear_interpolation2(data):
    empty_poses = np.where(data == 0)
    f = data.copy()
    for i in xrange(len(empty_poses[0])):
        pos = [p[i] for p in empty_poses]
        a_pos = map(lambda x: x - 1, pos)
        a_pos = np.array(a_pos)
        a_pos[np.where(a_pos < 0)] = 0
        b_pos = map(lambda x: x + 2, pos)  # +2 so it gets 3x3
        b_pos = np.array(b_pos)
        #logging.warning("Pos {0}, a_pos {1}, b_pos {2}, data.shape {3}"\
        #    .format(pos, a_pos, b_pos, data.shape))
        b_pos[np.where(b_pos > np.array(data.shape))] -= 1
        cell = data[a_pos[0]:b_pos[0], a_pos[1]:b_pos[1]]
        f[tuple(pos)] = cell[np.where(np.logical_and(cell != 0,
                             np.isnan(cell) == False))].mean()
        #logging.warning("Corrected position {0} to {1}".format(\
        #    pos, f[tuple(pos)]))
    return f


def make_cubic_interpolation(data):
    y, x = np.where(data !=0)
    Y, X = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    f = scint.griddata((x, y), data[y, x], (X, Y), 'cubic')
    #logging.warning('Done some shit')
    nans = np.where(np.isnan(f))
    #logging.warning('Got {0} nans after cubic'.format(len(nans[0])))
    f[nans] = 0
    f2 = make_linear_interpolation2(f)
    #logging.warning('Linear done')
    f[nans] = f2[nans]
    return f


def make_griddata_interpolation(plate, method='cubic'):

    points = np.where(np.logical_and(np.isnan(plate) == False,
                      plate > 0))

    values = plate[points]

    x_grid, y_grid = np.mgrid[0:plate.shape[0], 0:plate.shape[1]]

    res = scint.griddata(points, values, (x_grid, y_grid), method=method)

    if np.isnan(res).sum() > 0:

        methods = [(0, 'cubic'), (1, 'linear'), (2, 'nearest')]
        method_int_dict = {m[1]: m[0] for m in methods}
        int_method_dict = {m[0]: m[1] for m in methods}
        mval = method_int_dict[method]

        if mval < 2:

            method = int_method_dict[mval + 1]

            res = make_griddata_interpolation(res, method=method)

        else:

            logging.warning("There are still empty positions," +
                            " it should not be possible!")

    return res


#NEWEST PRECOG RATES
def get_empty_data_structure(plate, max_row, max_column, depth=None, default_value=48):

    def _allowed_size(val):
        ok = (8, 12, 16, 24, 32, 48, 64)
        if val in ok:
            return val
        else:
            for i in range(len(ok)):
                if ok[i] < val:
                    break

            if i == 0:
                return ok[0]
            else:
                return ok[i]

    data = np.array([None] * (plate + 1), dtype=np.object)

    for p in xrange(plate + 1):
        if depth is None:
            if max_row[p] > 0 and max_column[p] > 0:
                data[p] = np.ones((_allowed_size(max_row[p]),
                                   _allowed_size(max_column[p]))) * default_value
            else:
                data[p] = np.array([])
        else:
            if max_row[p] > 0 and max_column[p] > 0:
                data[p] = np.ones((_allowed_size(max_row[p]),
                                   _allowed_size(max_column[p]),
                                   depth)) * default_value
            else:
                data[p] = np.array([])

    return data


def get_default_surface_matrix(data):

    surface_matrix = {}
    for p in xrange(len(data)):
        surface_matrix[p] = [[0, 1], [0, 0]]

    return surface_matrix


def get_norm_surface_origin(data, p, cur_phenotype, norm_surface,
                            surface_matrix=None):

    if surface_matrix is None:
        surface_matrix = get_default_surface_matrix(data)

    original_norm_surface = norm_surface.copy()

    for x in xrange(data[p].shape[0]):
        for y in xrange(data[p].shape[1]):
            if surface_matrix[p][x % 2][y % 2] == 1:
                if np.isnan(data[p][x, y, cur_phenotype]) or data[p][x, y, cur_phenotype] == 48:
                    logging.warning(
                        "Discarded grid point ({0},{1})".format(x, y) +
                        " on plate {0} because it {1}.".format(
                            p, ["had no data", "had no growth"][
                                data[p][x, y, cur_phenotype] == 48]))
                else:
                    norm_surface[x, y] = data[p][x, y, cur_phenotype]

                original_norm_surface[x, y] = np.nan
                #print x, y

    return norm_surface, original_norm_surface


#USING 1/4th AS NORMALISING GRID
def get_norm_surface(data, cur_phenotype, sigma=3, only_linear=False,
                     surface_matrix=None):

    if surface_matrix is None:
        surface_matrix = get_default_surface_matrix(data)

    norm_surfaces = np.array([plate[...,cur_phenotype].copy()*0
                              for plate in data])
    norm_vals = []

    for p in range(len(data)):

        if len(data[p].shape) == 3:
            #exp_pp = map(lambda x: x/2, data[p].shape)
            #X, Y = np.mgrid[0:exp_pp[0],0:exp_pp[1]]*2
            norm_surface = norm_surfaces[p]
            #norm_surface[X, Y] = data[p][X, Y]
            #m = norm_surface[X, Y].mean()#.mean(1)
            #sd = norm_surface[X, Y].std()#.std(1)

            norm_surface, original_norm_surface = get_norm_surface_origin(
                data, p, cur_phenotype, norm_surface, surface_matrix)

            original_norms = norm_surface[
                np.logical_and(norm_surface != 0,
                               np.isnan(norm_surface) == False)]

            norm_vals.append(original_norms)
            m = original_norms.mean()
            #v = original_norms.var()
            sd = original_norms.std()

            logging.info(
                'Plate {0} has size {1} and {2} grid positions.'.format(
                    p, data[p].shape[0] * data[p].shape[1],
                    len(np.where(norm_surface != 0)[0])))
            logging.info(
                'High-bound outliers ({0}) on plate {1}'.format(
                    len(np.where(m + sigma * sd < original_norms)[0]), p))

            norm_surface[np.where(m + sigma * sd < norm_surface)] = 0

            logging.info(
                'Low-bound outliers ({0}) on plate {1}'.format(
                    len(np.where(m - sigma * sd > original_norms)[0]), p))

            norm_surface[np.where(m - sigma * sd > norm_surface)] = 0

            if only_linear:
                norm_surface = make_linear_interpolation2(norm_surface)
            else:
                norm_surface = make_griddata_interpolation(norm_surface)
                    # make_cubic_interpolation(norm_surface)

            #norm_surface[np.where(np.isnan(original_norm_surface))] = np.nan

            norm_surfaces[p] = norm_surface
            logging.info('Going for next plate')
        else:
            norm_surfaces[p] = np.array([])
            norm_vals.append(None)

    return norm_surfaces, np.array(norm_vals)


#NEW Luciano-format (And newest)
def get_data_from_file(file_path):
    try:
        fs = open(file_path, 'r')
    except:
        logging.error("Could not open file, no data was loaded")
        return None

    old_row = -1
    max_row = [-1, -1, -1, -1]
    max_column = [-1, -1, -1, -1]
    max_plate = 0
    plate = 0
    data_store = {}
    meta_data_store = {}

    for line in fs:
        try:

            l_tuple = [i.strip('"') for i in line.strip("\n").split("\t")]

            position, s_name, gt_anchors, value1, value2, value3 = l_tuple[:6]

            try:
                plate, row_column = map(lambda x: x.strip(":"),
                                        position.split(" "))
                row, column = row_column.split("-")
            except:
                plate, row_column = position.split(":", 1)
                row, column = row_column.split("-", 1)

            plate, row, column = map(int, (plate, row, column))
            if max_row[plate] < row:
                max_row[plate] = row
            if max_column[plate] < column:
                max_column[plate] = column
            if max_plate < plate:
                max_plate = plate

            try:
                value1 = float(value1.replace(",", ".").strip())
            except:
                value1 = np.nan
            try:
                value2 = float(value2.replace(",", ".").strip())
            except:
                value2 = np.nan
            try:
                value3 = float(value3.replace(",", ".").strip())
            except:
                value3 = np.nan

            data_store[(plate, row, column)] = (value1, value2, value3)

            gt_anchors = gt_anchors.strip()[1:-1]
            gt_anchors = gt_anchors.split(",")
            try:
                """
                gt_anchors = map(lambda x: x.split("-"), gt_anchors)
                data_rates = map(float, [d[0] for d in gt_anchors])
                """
                gt_times = map(float, gt_anchors)
                meta_data_store[(plate, row, column)] = (s_name, gt_times)
            except:
                meta_data_store[(plate, row, column)] = (s_name, [])

                #logging.warning("Data rows should be\nPOSITION\tSTRAIN_NAME\tGT_ANCHORS\tPHENOTYPE1\tPHENOTYPE2\tPHENOTYPE3")
                #logging.warning("Position {0} has unrecognized meta-data format: {1}".\
                #    format((plate,row,column), data))
        except:
            try:
                position, data = line.split(" ", 1)
                row, column = map(int, position.split("-"))
                if row < old_row:
                    plate += 1
                data = eval(data)
                if len(data) == 5:
                    data_store[(plate, row, column)] = data
                    if row > max_row:
                        max_row = row
                    if column > max_column:
                        max_column = column
                old_row = row
                logging.warning("Data rows should be\nPOSITION\tSTRAIN_NAME\tGT_ANCHORS\tPHENOTYPE1\tPHENOTYPE2\tPHENOTYPE3")

            except:
                logging.warning("BAD data row: '%s'" % line)

    fs.close()
    data = get_empty_data_structure(max_plate, max_row, max_column, depth=3, default_value=48)
    for k, v in data_store.items():
        try:
            data[k[0]][k[1], k[2]] = v
        except:
            logging.error(
                "Got unexpected index {0} when data-shape is {1}, plate {2}, values {3}".format(
                    k, data.shape, data[k[0]].shape, v))
            return None

    return {'data': data, 'meta-data': meta_data_store}


def get_outlier_phenotypes(data, alpha=3, cur_phenotype=0):

    suspects = []

    for i, p in enumerate(data):
        if len(p.shape) == 3:
            p = p[:, :, cur_phenotype]
            d = p[np.isnan(p) != True]
            #print d.size, alpha * d.mean(), d.std(), p.shape
            #print np.abs(p - d.mean())
            #print np.logical_and(np.abs(p - d.mean()) > alpha * d.std(), np.isnan(p) == False).sum()
            coords = np.where(np.logical_and(abs(p - d.mean()) > alpha * d.std(),
                                             np.isnan(p) == False))

            suspects += zip([i] * len(coords[0]), coords[0], coords[1])

    return suspects


def get_dubious_phenotypes(meta_data_store, plate, max_row, max_column):

    if meta_data_store == {}:
        logging.info("No meta-data to analyse")
        return None

    data = get_empty_data_structure(plate, max_row, max_column)
    dubious = 0
    dubious2 = 0
    st_pt = 0
    ln_pts = 7
    side_buff = 1
    for loc, measures in meta_data_store.items():

        times = np.array(measures[0])
        OD = np.array(measures[1])
        f_OD = OD[st_pt:ln_pts][times[st_pt:ln_pts].argsort()[-1 - side_buff]] / \
            OD[st_pt:ln_pts][times[st_pt:ln_pts].argsort()[side_buff]]
        d_t = times[times[st_pt:ln_pts].argsort()[-1 - side_buff]] -\
            times[times[st_pt:ln_pts].argsort()[side_buff]]
        data[loc[0]][loc[1:]] = d_t * np.log(2) / np.log2(f_OD)

        if (times[st_pt:ln_pts].min() in times[st_pt:ln_pts][-2:] and
                times[st_pt:ln_pts].max() in times[st_pt:ln_pts][-2:]):

            logging.warning("Encountered dubious thing".format(d_t, f_OD))
            dubious += 1

        if (times[st_pt:ln_pts].min() in times[st_pt:ln_pts][:2] or
                times[st_pt:ln_pts].max() in times[st_pt:ln_pts][:2]):

            dubious2 += 1

    logging.info(
        "Found {0} dubious type 1 and {1} type 2 dubious".format(
            dubious, dubious2))

    return dubious, dubious2


def show_heatmap(data, cur_phenotype, plate_texts, plate_text_pattern, vlim=(0, 48)):
    #PLOT A SIMPLE HEAT MAP
    fig = plt.figure()
    nplates = len(data)
    rows = np.ceil(nplates / 2.0)
    columns = np.ceil(nplates / 2.0)
    for p in xrange(nplates):
        if len(data[p].shape) >= 2:
            ax = fig.add_subplot(
                rows, columns, p + 1, title=plate_text_pattern.format(
                    plate_texts[p]))
            if len(data[p].shape) == 3:
                pdata = data[p][..., cur_phenotype]
            else:
                pdata = data[p]
            im = ax.imshow(pdata, vmin=vlim[0],
                      vmax=vlim[1], interpolation="nearest",
                      cmap=plt.cm.jet)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(im, cax=cax)

            #ax.set_axis_off()

        else:
            logging.warning("Plate {0} has no values (shape {1})".format(p, data[p].shape))

    fig.tight_layout()
    fig.show()
    #PLOT DONE


def get_interactive_header(header, width=60):

    print "\n"
    print header.center(width)
    print ("-" * len(header)).center(width)
    print "\n"


def get_interactive_info(info_list, margin=6):

    for info in info_list:
        print " " * margin + info

    print "\n"


def get_interactive_labels(header, labels, meta_data):

    get_interactive_header(header)

    plate_labels = {}

    try:
        print "\t{0}\n".format(meta_data['desc'])
    except:
        pass

    for i in xrange(labels):
        plate_labels[i] = raw_input("Label for Plate {0}: ".format(i))

    return plate_labels


def get_interactive_norm_surface_matrix(data, cur_grids):

    get_interactive_header("This sets up the normalisation grid per plate")

    get_interactive_info([
        "Each 2x2 square of colonies is expected to have both",
        "norm-grid positions and experiment posistions",
        "1 means it's a norm-grid position",
        "0 means it's an experiment position"])

    norm_surface_matrices = {}
    default_grid = [[0, 1], [0, 0]]
    for p in xrange(len(data)):
        norm_matrix = [None, None]
        if cur_grids is not None and cur_grids[p] is not None:
            cur_grid = cur_grids[p]
        else:
            cur_grid = default_grid

        print "\n-- For Plate {0} --".format(p)
        print "\nCurrent format is:"
        print "{0} {1}\n{2} {3}".format(*itertools.chain(*cur_grid))
        for row in xrange(2):
            r = raw_input(
                "\n** ROW {0} ('Press <ENTER> to keep current') :".format(
                    ["ONE", "TWO"][row > 0]))
            try:
                norm_matrix[row] = map(int, r.split(" ", 1))
            except:
                logging.info("Current is used for row {0}".format(row + 1))
                norm_matrix[row] = cur_grid[row]

        norm_surface_matrices[p] = norm_matrix

    return norm_surface_matrices


def get_normalised_values(data, cur_phenotype, surface_matrices=None, do_log=None):
    #THIS RETURNS LSC VALUES!

    norm_surface, norm_vals = get_norm_surface(
        data, cur_phenotype, surface_matrix=surface_matrices)

    normed_data = np.array([p[..., cur_phenotype].copy() * np.nan for p in data])

    logging.info(
        "The norm-grid means where {0} (surface mean was {1})".format(
            [p.mean() for p in norm_vals],
            [np.mean(p[np.where(np.isnan(p) == False)]) for p in norm_surface]))

    logging.info(
        "Zero positions in normsurface: {0}".format([(p == 0).sum() for p in norm_surface]))

    for p in xrange(len(data)):
        if do_log:
            normed_data[p] = (np.log2(data[p][..., cur_phenotype]) - np.log2(norm_surface[p]))  # + norm_means[p]
        else:
            normed_data[p] = (data[p][..., cur_phenotype] - norm_surface[p])  # + norm_means[p]

    return normed_data, norm_vals


def get_experiment_results(data, surface_matrices=None, cur_phenotype=0):

    nplates = len(data)
    exp_pp = map(lambda x: map(lambda y: y / 2, x.shape[:-1]), data)
    e_mean = np.array([None] * nplates, dtype=np.object)
    e_data = np.array([None] * nplates, dtype=np.object)
    e_sd = np.array([None] * nplates, dtype=np.object)

    """
    e_max = 0
    e_min = 0
    e_sd_max = 0
    e_sd_min = 0
    """
    if surface_matrices is None:
        surface_matrices = get_default_surface_matrix(data)

    #CALC EXPERIMENTS
    for p in range(len(data)):

        if len(data[p].shape) == 3:
            exp_filter = np.array(surface_matrices[p]) == 0
            exp_data = np.zeros(
                exp_pp[p] + [exp_filter.sum()], dtype=np.float64)
            exp_mean = np.zeros(exp_pp[p], dtype=np.float64)
            exp_sd = np.zeros(exp_pp[p], dtype=np.float64)
            for x in range(exp_pp[p][0]):
                for y in range(exp_pp[p][1]):
                    cell = data[p][x * 2:x * 2 + 2, y * 2:y * 2 + 2, cur_phenotype]
                    cell_data = cell[np.where(exp_filter)]
                    exp_data[x, y, :] = cell_data
                    exp_mean[x, y] = cell_data[
                        np.where(np.isnan(cell_data) == False)].mean()
                    exp_sd[x, y] = cell_data[
                        np.where(np.isnan(cell_data) == False)].std()

                #logging.warning("Plate {0}, row {1}".format(p, x))
            """
            if e_max < exp_mean.max():
                e_max = exp_mean.max()
            if e_min > exp_mean.min():
                e_min = exp_mean.min()
            if e_sd_max < exp_sd.max():
                e_sd_max = exp_sd.max()
            if e_sd_min > exp_sd.min():
                e_sd_min = exp_sd.min()
            """
            e_mean[p] = exp_mean
            e_sd[p] = exp_sd
            e_data[p] = exp_data
        else:
            e_mean[p] = np.array([])
            e_sd[p] = np.array([])
            e_data[p] = np.array([])

    return e_mean, e_sd, e_data


class Interactive_Menu():

    def __init__(self):

        self._menu = {
            '1A': 'Load data',
            '1B': 'Select Phenotype To Work With',
            '1C': 'Name Phenotypes',
            'Q1': 'Weed out superbad curves automatically (requires xml-file)',
            'Q2': 'Inspect/remove outliers (requires xml-file)',
            'Q3': 'Inspect/remove based on bad friends (requires xml-file)',
            'Q4': 'Manual remove stuff',
            '2A': 'Set normalisation grid position',
            '2B': 'Normalise data',
            '2C': 'Calculate experiments',
            'R1': 'Re-map positions -- rotate',
            'R2': 'Re-map positions -- move',
            'R3': 'Re-map positions -- flip',
            'P1': 'Set plot plate names',
            'P2': 'Show heatmap(s) of original data',
            'P3': 'Show heatmap(s) of normalised data',
            'P4': 'Show heatmap(s) of per experment data',
            'U': 'Undo back to last save',
            'S': 'Save all available data (will overwrite files if they exist)',
            'T': 'Terminate! (Quit)'}
        self._menu_order = ('1A', '1B', '1C', 'Q1', 'Q2', 'Q3', 'Q4', '2A', '2B', '2C', 'R1',
                            'R2', 'R3', 'P1', 'P2', 'P3', 'P4', 'U', 'S', 'T')

        self._enabled_menus = {}
        self.set_start_menu_state()

        self._cur_phenotype = 0
        self._experiments = None
        self._experiments_data = None
        self._experiments_sd = None
        self._file_dir_path = None
        self._file_name = None
        self._file_path = None
        self._xml_file = None
        self._xml_connection = [0, [0, 0]]
        self._grid_surface_matrices = None
        self._plate_labels = None
        self._original_phenotypes = None
        self._LSC_phenotypes = None
        self._normalisation_vals = None
        self._original_meta_data = None
        self._data_shapes = None
        self._is_logged = False
        self._is_normed = None
        self._phenotype_names = None
        self._meta_data = None

    def set_start_menu_state(self):
        for k in self._menu.keys():
            if k in ['1A', 'T']:
                self._enabled_menus[k] = True
            else:
                self._enabled_menus[k] = False

    def set_new_file_menu_state(self):
        self.set_start_menu_state()
        self.set_enable_menu_items(['1B', '1C', 'Q1', 'Q2', 'Q3', 'Q4',
                                    '2A', '2B', 'P1',
                                    'R1', 'R2', 'R3', 'U', 'S'])

    def set_enable_menu_items(self, itemlist):

        for i in itemlist:
            self._enabled_menus[i] = True

    def set_disable_menu_items(self, itemlist):

        for i in itemlist:
            self._enabled_menus[i] = False

    def set_enable_menu_plots(self):

        disables = []
        enables = []

        if self._plate_labels is not None:

            if self._original_phenotypes is not None:

                enables.append("P2")
            else:
                disables.append("P2")

            if (self._is_normed[self._cur_phenotype] and
                    self._LSC_phenotypes is not None):

                enables.append("P3")
            else:
                disables.append("P3")

            if (self._is_normed[self._cur_phenotype] and
                    self._experiments is not None):

                enables.append("P4")
            else:
                disables.append("P4")

        self.set_enable_menu_items(enables)
        self.set_disable_menu_items(disables)

    def print_menu(self):

        get_interactive_header(
            "Menu{0}".format([" ({0})".format(self._file_path), ""][
            self._file_path is None]))

        try:
            print "\t{0}\n".format(self._meta_data['desc'])
        except:
            pass

        print "\t### PHENOTYPE {0} ###\n".format(
            self._phenotype_names is None and self._cur_phenotype + 1 or
            self._phenotype_names[self._cur_phenotype])

        old_k = None
        for k in self._menu_order:

            if self._enabled_menus[k]:
                if old_k is not None and k[0] != old_k[0]:
                    print "  --"
                print " " * 2 + k + " " * (6 - len(k)) + self._menu[k]
                old_k = k

        print "\n"

    def get_answer(self):

        answer = str(raw_input("Run option: ")).upper()

        if answer in self._menu.keys() and self._enabled_menus[answer]:
            return answer
        else:
            logging.info("{0} is not a valid option".format(answer))

            return None

    def set_plate_labels(self):

        self._plate_labels = get_interactive_labels(
            "Setting plate names", self._original_phenotypes.shape[0], self._meta_data)
        logging.info("New labels set")
        self.set_enable_menu_plots()

    def set_nan_from_list(self, nan_list):

        for pos in nan_list:

            self._original_phenotypes[pos[0]][pos[1], pos[2], self._cur_phenotype] = np.nan


        self._is_normed[self._cur_phenotype] = False
        self.set_phenotype_is_normed_state()

    def set_xml_file(self):

        if self._file_path is None:
            return

        file_path = self._file_path.split(".")
        pickle_data_file = ".".join(file_path[:-1]) + ".data.pickle"
        pickle_scantimes_file = ".".join(file_path[:-1]) + "scantimes.pickle"
        pickle_metadata_file = ".".join(file_path[:-1]) + ".metadata.pickle"
        pickle_matrix_file = ".".join(file_path[:-1]) + ".matrix.pickle"
        pickle_current_phenotype_file = ".".join(file_path[:-1]) + ".cur_pheno.pickle"
        pickle_is_normed_file = ".".join(file_path[:-1]) + ".isnormed.pickle"
        pickle_pheno_names_file = ".".join(file_path[:-1]) + ".pheno_names.pickle"

        if os.path.isfile(pickle_data_file) and os.path.isfile(pickle_metadata_file):

            data = pickle.load(open(pickle_data_file, 'rb'))
            meta_data = pickle.load(open(pickle_metadata_file, 'rb'))
            scan_times = pickle.load(open(pickle_scantimes_file, 'rb'))

            self._xml_file = r_xml.XML_Reader(data=data, meta_data=meta_data,
                                              scan_times=scan_times)
            self._meta_data = meta_data
            if os.path.isfile(pickle_matrix_file):
                self._grid_surface_matrices = pickle.load(open(
                    pickle_matrix_file, 'rb'))
            if os.path.isfile(pickle_current_phenotype_file):
                self._cur_phenotype = pickle.load(open(
                    pickle_current_phenotype_file, 'rb'))

            """
            if os.path.isfile(pickle_is_normed_file):
                self._is_normed = pickle.load(open(
                    pickle_is_normed_file, 'rb'))
            """

            if os.path.isfile(pickle_pheno_names_file):
                self._phenotype_names = pickle.load(open(
                    pickle_pheno_names_file, 'rb'))

        else:

            file_path = ".".join(file_path[:-1]) + ".xml"
            good_guess = True
            try:
                fs = open(file_path, 'r')
                fs.close()
                print "Maybe this is the file: '{0}'?".format(file_path)
            except:
                good_guess = False
                file_path = file_path.split(os.sep)
                file_path = os.sep.join(file_path[:-1])
                print "Maybe this is the directory of the file: '{0}'".format(file_path)
            answer = str(raw_input("The path to the xml-file {0}: ".format(
                ["", "(Press enter to use suggested file)"][good_guess])))

            if answer != "":
                file_path = answer

            logging.info("This may take a while...")
            self._xml_file = r_xml.XML_Reader(file_path)
            pickle.dump(self._xml_file.get_data(), open(pickle_data_file, 'wb'))
            pickle.dump(self._xml_file.get_meta_data(), open(pickle_metadata_file, 'wb'))
            pickle.dump(self._xml_file.get_scan_times(), open(pickle_scantimes_file, 'wb'))

        self.set_enable_menu_plots()

    def review_positions_for_deletion(self, suspect_list=None):

            if self._xml_file is None:
                self.set_xml_file()

            if self._xml_file.get_loaded():

                if suspect_list is None:
                    suspect_list = get_outlier_phenotypes(
                        self._original_phenotypes,
                        cur_phenotype=self._cur_phenotype)

                remove_list = []

                shapes = self._xml_file.get_shapes()

                measurement = 0

                if shapes[0][-1][-1] > 1:

                    print "There are multiple measures for each colony, which to you want to see?\n"
                    print "Value example (it is all you get):"
                    d = self._xml_file.get_colony(shapes[0][0], 0, 0)
                    print list(d[-1, :])

                    try:

                        measurement = int(raw_input(
                            'Which do you want (0 - {0})? '.format(
                            d.shape[-1])))

                    except:

                        measurement = 0

                fig = plt.figure()

                fig.show()

                for s in suspect_list:

                    if not plt.fignum_exists(fig.number):
                        fig = plt.figure()
                        fig.show()

                    fig.clf()

                    try:
                        phenotypes = (
                            "{0}: {1}".format(
                                self._original_meta_data[s][0],
                                self._original_phenotypes[
                                    s[0]][s[1], s[2], self._cur_phenotype]),)
                    except:
                        phenotypes = (
                            str(self._original_phenotypes[
                                s[0]][s[1], s[2], self._cur_phenotype]),)

                    print phenotypes, s
                    fig = r_xml.plot_from_list(
                        self._xml_file, [s], fig=fig, measurement=measurement,
                        phenotypes=phenotypes,
                        ax_title="Plate {0} Pos {1}: Phenotype {2}".format(
                            s[0], s[1:],
                            (self._phenotype_names is None and self._cur_phenotype or
                             self._phenotype_names[self._cur_phenotype])))

                    fig.show()
                    if str(raw_input("Is this a bad curve (y/N)?")).upper() == "Y":

                        remove_list.append(s)

                logging.info("{0} outliers will be removed".format(len(remove_list)))

                self.set_nan_from_list(remove_list)

    def load_file(self, file_path):

        if os.sep not in file_path:
            file_path = ".{0}{1}".format(os.sep, file_path)

        file_contents = get_data_from_file(file_path)

        if file_contents is not None:

            self._file_path = file_path
            self._file_name = ".".join(file_path.split(os.sep)[-1].split(".")[:-1])
            self._file_dir_path = os.sep.join(file_path.split(os.sep)[:-1])

            self._original_phenotypes = file_contents['data']
            self._original_meta_data = file_contents['meta-data']

            mins = []
            maxs = []

            for p in xrange(self._original_phenotypes.shape[0]):
                if len(self._original_phenotypes[p].shape) == 3:
                    mins.append(self._original_phenotypes[p].min(axis=0).min(axis=0))
                    maxs.append(self._original_phenotypes[p].max(axis=0).max(axis=0))
                else:
                    mins.append("N/A")
                    maxs.append("N/A")

            logging.info(
                "File had {0} plates,".format(
                    self._original_phenotypes.shape[0]) +
                "and phenotypes ranged from\n{0}\nto\n{1}\nper plate.".format(
                    mins, maxs))

            self.set_new_file_menu_state()
            self._plate_labels = {}
            for i in xrange(self._original_phenotypes.shape[0]):
                self._plate_labels[i] = str(i)
            self._is_normed = [False] * self._original_phenotypes.shape[0]
            logging.info("Temporary plate labels set")
            self.set_enable_menu_plots()
            self.set_xml_file()

            return True

        else:
            return False

    def set_phenotype_is_normed_state(self):

        if self._is_normed[self._cur_phenotype]:

            plates = [p[..., self._cur_phenotype] for p in self._LSC_phenotypes]

            self._LSC_min = min([p[np.isnan(p) == False].min() for
                                 p in plates])
            self._LSC_max = min([p[np.isnan(p) == False].max() for
                                 p in plates])

            self._experiments, self._experiments_sd, self._experiments_data =\
                get_experiment_results(self._LSC_phenotypes,
                                       self._grid_surface_matrices,
                                       self._cur_phenotype)

            #self.set_enable_menu_items(["2C"])

            logging.info(
                "These values are indicative of the general" +
                " quality of the experiment (lower better " +
                "(variance of the grid before normalisation)):\n{0}".format(
                    [p.var() for p in self._normalisation_vals[self._cur_phenotype]]))

        self.set_enable_menu_plots()

    def get_interactive_plate(self):

        print "From which plate?"
        if self._plate_labels is not None:
            for i, p in self._plate_labels.items():
                print " " * 2 + str(i) + " " * 5 + p
        else:
            print "(0 - {0})".format(self._original_phenotypes.shape[0] - 1)

        #INPUT AND DISCARD NON INTs
        try:
            plate = int(raw_input("> "))
        except:
            plate = None
            logging.warning("Not a valid plate number")

        #OUT OF RANGE CHECK
        if self._original_phenotypes.shape[0] <= plate or plate < 0:
            logging.warning("Out of bounds {0}".format(plate))
            plate = None

        return plate

    def do_task(self, task):

        if task == "1A":

            file_path = str(raw_input("The path to the file: "))
            if self.load_file(file_path) is False:

                logging.warning("Nothing changed...")

        elif task == "1B":

            phenotype = str(raw_input("What phenotype (1-3)? "))
            try:
                phenotype = int(phenotype)
                if self._original_phenotypes[0].shape[-1] >= phenotype:
                    self._is_normed[self._cur_phenotype] = False
                    self._cur_phenotype = phenotype - 1
                    logging.info("Phenotype changed to {0}".format(phenotype))
            except:
                pass

            self.set_enable_menu_plots()

        elif task == "1C":

            self._phenotype_names = []
            for i in range(3):
                phenotype_name = str(raw_input("Name of phenotype {0}: ".format(i+1)))
                self._phenotype_names.append(phenotype_name)

        elif task == "Q1":

            removal_list = []
            if self._xml_file is None or self._xml_file.get_loaded() is False:
                self.set_xml_file()

            if self._xml_file.get_loaded():

                #t = 0.6
                for p in xrange(self._original_phenotypes.shape[0]):
                    for c in xrange(self._original_phenotypes[p].shape[0]):
                        for r in xrange(self._original_phenotypes[p].shape[1]):

                            d = np.isnan(
                                self._xml_file.get_colony(p, c, r)).astype(
                                    np.int8)

                            if (1 - d.sum() / float(d.shape[0]) < 0.6 and
                                    np.isnan(
                                        self._original_phenotypes[
                                            p, c, r, self._cur_phenotype
                                        ]).any() is False):

                                removal_list.append((p, c, r))

                logging.info(
                    "Removed {0} positions because of superbadness".format(
                        len(removal_list)))

                self.set_nan_from_list(removal_list)

        elif task == "Q2":

            self.review_positions_for_deletion()

        elif task == "Q3":

            suspect_list = []

            for p in xrange(self._original_phenotypes.shape[0]):
                for c in xrange(0, self._original_phenotypes[p].shape[0], 2):
                    for r in xrange(0, self._original_phenotypes[p].shape[1], 2):

                        nans = np.isnan(self._original_phenotypes[p][
                            2 * c:2 * c + 2, 2 * r:2 * r + 2, self._cur_phenotype])

                        if nans.any():
                            pos = np.where(nans==False)
                            for pp in xrange(len(pos[0])):

                                suspect_list.append(
                                    (p, 2 * c + pos[0][pp],
                                     2 * r + pos[1][pp]))

                                #print self._original_phenotypes[p][2*c+pos[0][pp],2*r+pos[1][pp]]

            self.review_positions_for_deletion(suspect_list)

        elif task == "Q4":

            answer = "0"
            removal_list = []

            while answer not in ["", "A"]:

                for o in [
                        ('R', 'Remove a Row from a Plate (Per plate count: {0})'),
                        ('C', 'Remove a Column from a Plate (Per plate count: {0})'),
                        ('P', 'Remove an individual Position from a Plate'),
                        ('L', 'Look through and review the selected'),
                        ('A', 'Abort')]:

                    items = [(["-", ""][len(p.shape) == 2])
                             for p in self._original_phenotypes]

                    for i, it in enumerate(items):

                        if it == "":

                            items[i] = self._original_phenotypes[i].shape[
                                o[0] == 'C']

                    print " " * 2 + o[0] + " " * (6 - len(o[0])) + o[1].format(items)

                print "\nJust press enter when you are ready to remove all you have selected"

                answer = str(raw_input("What's your wish? ")).upper()

                if answer not in ["", "A", "L"] and answer in ['R', 'C', 'P']:

                    plate = self.get_interactive_plate()

                    if plate is not None:

                        row = 0
                        column = 0

                        if answer in ["R", "P"]:

                            row = int(
                                raw_input("Which row (0 - {0}): ".format(
                                    self._original_phenotypes[plate].shape[0] - 1)))

                            if row < 0 or row >= self._original_phenotypes[plate].shape[0]:
                                row = None

                        if answer in ["C", "P"]:

                            column = int(
                                raw_input("Which column (0 - {0}): ".format(
                                    self._original_phenotypes[plate].shape[1] - 1)))

                            if column < 0 or column >= self._original_phenotypes[plate].shape[1]:
                                column = None

                        if None in (row, column):

                            logging.warning("Index out of bounds")

                        else:

                            if answer == "P":

                                pos = (plate, row, column)
                                logging.info("{0} is now selected for removal".format(pos))
                                removal_list.append(pos)

                            elif answer in ["R", "C"]:

                                pos_list = []
                                for i in xrange(
                                        self._original_phenotypes[plate].shape[
                                            answer == "R"]):

                                    pos_list.append(
                                        (plate, [i, row][answer == "R"],
                                         [column, i][answer == "R"]))

                                logging.info("The following positions are not marked for removal {0}".format(pos_list))

                                removal_list += pos_list

                elif answer == "L":

                    self.review_positions_for_deletion(suspect_list=removal_list)
                    logging.info("The suspect list that you had compiled is now empty")
                    removal_list = []

                elif answer == "" and len(removal_list) > 0:

                    answer = str(
                        raw_input("This will remove {0} colonies,".format(
                            len(removal_list)) + " are you 112% sure (y/N)? "
                        )).upper()

                    if answer == "Y":
                        self.set_nan_from_list(removal_list)
                        removal_list = []

        elif task == "2A":

            self._grid_surface_matrices = get_interactive_norm_surface_matrix(
                self._original_phenotypes,
                self._grid_surface_matrices)
            logging.info("New normalisation grid set!")

        elif task == "2B":

            if self._LSC_phenotypes is None:

                self._LSC_phenotypes = [np.zeros(p.shape) for p in
                    self._original_phenotypes]

                self._normalisation_vals = [None] * 3

            LSC_phenotypes, normalisation_vals = \
                get_normalised_values(
                    self._original_phenotypes,
                    self._cur_phenotype,
                    self._grid_surface_matrices,
                    do_log=(self._is_logged is False))

            for i in range(len(self._LSC_phenotypes)):

                self._LSC_phenotypes[i][..., self._cur_phenotype] = LSC_phenotypes[i]

            self._normalisation_vals[self._cur_phenotype] = normalisation_vals

            self._is_normed[self._cur_phenotype] = True
            self.set_phenotype_is_normed_state()

        elif task == "R1":

            plate = self.get_interactive_plate()

            if plate is not None:
                print "Should the (0,0) position be:"

                for i, p in enumerate(
                        [
                        #"({0},0)".format(self._original_phenotypes[plate].shape[0]-1),
                        "(0,{0})".format(
                            self._original_phenotypes[plate].shape[0] - 1),
                        #"(0,{0})".format(self._original_phenotypes[plate].shape[1]-1),
                        "({0},{1})".format(
                            self._original_phenotypes[plate].shape[0] - 1,
                            self._original_phenotypes[plate].shape[1] - 1),
                        #"({0},{1})".format(self._original_phenotypes[plate].shape[1]-1,
                        #self._original_phenotypes[plate].shape[0]-1)
                        "({0},0)".format(
                            self._original_phenotypes[plate].shape[1] - 1)
                        ]):

                    print " " * 2 + str(i) + " " * 5 + p

                rotation = str(raw_input("Select way to rotate/flip your data or press enter to abort: "))

                if rotation in ["0", "1", "2"]:
                    self._original_phenotypes[plate] = np.rot90(self._original_phenotypes[plate])
                if rotation in ["0", "1"]:
                    self._original_phenotypes[plate] = np.rot90(self._original_phenotypes[plate])
                if rotation in ["0"]:
                    self._original_phenotypes[plate] = np.rot90(self._original_phenotypes[plate])

                self._xml_connection[0] = (
                    self._xml_connection[0] + (3 - int(rotation))) % 4

        elif task == "R2":

            plate = self.get_interactive_plate()

            if plate is not None:
                new_pos_str = str(raw_input("The current (0,0) position should be: "))

                bad_pos = False
                try:
                    new_pos = map(int, eval(new_pos_str))
                except:
                    bad_pos = True
                if not bad_pos:
                    if len(new_pos) != 2:
                        bad_pos = True

                if bad_pos:
                    logging.error("Could not understand you input")
                else:
                    m = new_pos[0]
                    if m > 0:
                        self._original_phenotypes[plate][m:, :] = self._original_phenotypes[plate][:-m, :]
                        self._original_phenotypes[plate][:m, :] = np.nan
                    elif m < 0:
                        self._original_phenotypes[plate][:m, :] = self._original_phenotypes[plate][-m:, :]
                        self._original_phenotypes[plate][m:, :] = np.nan
                    m = new_pos[1]
                    if m > 0:
                        self._original_phenotypes[plate][:, m:] = self._original_phenotypes[plate][:, :-m]
                        self._original_phenotypes[plate][:, :m] = np.nan
                    elif m < 0:
                        self._original_phenotypes[plate][:, :m] = self._original_phenotypes[plate][:, -m:]
                        self._original_phenotypes[plate][:, m:] = np.nan

                    logging.info("It has been moved and unkown places filled with nan")
                    logging.info("You need to reload the data-set if you want to undo")
                    logging.info("Also remember to re-run normalisation etc.")

                    self._xml_connection[1][0] += new_pos[0]
                    self._xml_connection[1][1] += new_pos[1]

        elif task == "R3":

            plate = self.get_interactive_plate()

            if plate is not None:

                flip_dim = str(
                    raw_input(
                        "Which dimension should be flipped? " +
                        "(0 (size: {0}) / 1 (size: {1}) / Abort (anything else)): ".format(
                            self._original_phenotypes[plate].shape[0],
                            self._original_phenotypes[plate].shape[1])))

                if flip_dim in ['0', '1']:

                    flip_dim = int(flip_dim)

                    dim_size = self._original_phenotypes[plate].shape[flip_dim]

                    if flip_dim == 0:
                        self._original_phenotypes[plate] = \
                            self._original_phenotypes[plate][
                                np.arange(dim_size - 1, -1, -1), :]
                    elif flip_dim == 1:
                        self._original_phenotypes[plate] = \
                            self._original_phenotypes[plate][
                                :, np.arange(dim_size - 1, -1, -1)]

                    logging.info("Flip is done, but you should have done this the last thing you do before normalising!")
                else:

                    logging.info("No flip done!")

        elif task == "P1":

            self.set_plate_labels()

        elif task == "P2":

            show_heatmap(
                self._original_phenotypes,
                self._cur_phenotype,
                self._plate_labels,
                "Original Phenotypes (Plate {0})",
                vlim=(0, [48, 5.58][self._is_logged]))

        elif task == "P3":

            show_heatmap(
                self._LSC_phenotypes,
                self._cur_phenotype,
                self._plate_labels,
                "LSC (Plate {0})",
                vlim=(self._LSC_min, self._LSC_max))

        elif task == "P4":

            show_heatmap(
                self._experiments,
                self._cur_phenotype,
                self._plate_labels,
                "Mean LSC (Plate {0})",
                vlim=(self._LSC_min, self._LSC_max))

            show_heatmap(
                self._experiments_sd,
                self._cur_phenotype,
                self._plate_labels,
                "LSC std (Plate {0})",
                vlim=(self._LSC_min, self._LSC_max))

        elif task == "U":

            try:
                undo = (raw_input(
                    "Are you sure you want to undo all unsaved work (y/N)?")[0].upper() ==
                    "Y")

            except:
                undo = False

            if undo:
                self.set_xml_file()

        elif task == "S":

            if self._original_phenotypes is not None:
                header = "Saving non-normed data"
                if self._save(
                        self._original_phenotypes, header,
                        save_as_np_array=True,
                        file_guess="_original.npy"):

                    logging.info("Data saved!")

                else:

                    logging.warning("Could not save data, probably path is not valid")

            file_path = self._get_save_file_guess(file_guess=".cur_pheno.pickle")
            try:
                pickle.dump(self._cur_phenotype, open(file_path, 'wb'))
            except:
                logging.warning("Could not save current phenotype state")

            file_path = self._get_save_file_guess(file_guess=".matrix.pickle")
            try:
                pickle.dump(self._grid_surface_matrices, open(file_path, 'wb'))
            except:
                logging.warning("Could not save Grid Lawn position")

            """
            file_path = self._get_save_file_guess(file_guess=".isnormed.pickle")
            try:
                pickle.dump(self._is_normed, open(file_path, 'wb'))
            except:
                logging.warning("Could not save normalisation status")
            """

            file_path = self._get_save_file_guess(file_guess=".pheno_names.pickle")
            try:
                pickle.dump(self._phenotype_names, open(file_path, 'wb'))
            except:
                logging.warning("Could not save normalisation status")

            """
            if self._original_meta_data is not None:

                header = "Saving metadata"
                self._set_save_intro(header)
                file_path = self._get_save_file_guess(file_guess=".strain_metadata.pickle")
                try:
                    pickle.dump(self._original_meta_data, open(file_path, 'wb'))
                    logging.info("Data saved!")
                except:
                    logging.warning("Could not save meta-data")
            """

            if self._normalisation_vals is not None:

                for t in range(1):
                    dtype = ('npy',)[t]
                    header = "Saving values from normalisation grid ({0})".format(dtype)
                    if self._save(
                            np.array(self._normalisation_vals), header,
                            save_as_np_array=True,
                            file_guess="_norm_array.{0}".format(dtype)):

                        logging.info("Data saved!")

                    else:

                        logging.warning("Could not save data, probably path is not valid")

            if self._LSC_phenotypes is not None:
                for task in ["S1", "S3"]:

                    header = "Saving LSC phenotypes as {0}".format(
                        ["csv", "numpy-array"][task == "S3"])

                    if self._save(
                            self._LSC_phenotypes, header,
                            save_as_np_array=(task == "S3"),
                            file_guess='_LSC.{0}'.format(
                                ['csv', 'npy'][task == 'S3'])):

                        logging.info("Data saved!")

                    else:

                        logging.warning("Could not save data, probably path is not valid")

            if self._experiments_data is not None:
                for task in ["S2", "S4"]:

                    header = "Saving LSC per experiment as {0}".format(
                        ["csv", "numpy-array"][task == "S4"])

                    if self._save(
                            self._experiments_data, header,
                            save_as_np_array=(task == "S4"),
                            file_guess='_LSC_experment.phenotype_{0}.{1}'.format(
                                self._cur_phenotype + 1,
                                ['csv', 'npy'][task == 'S4'])):

                        logging.info("Data saved!")

                    else:

                        logging.warning("Could not save data, probably path is not valid")

    def _get_save_file_guess(self, file_guess=None):

        if file_guess is not None:
            file_path = self._file_dir_path + os.sep + self._file_name + file_guess
            if str(raw_input("Maybe this is a good place:\n'{0}'\n(Y/n)?".format(file_path))).upper() == "N":
                file_path = None

        if file_path is None:
            file_path = str(raw_input("Save-path (with file name): "))

        return file_path

    def _set_save_intro(self, header):

        get_interactive_header(header)
        get_interactive_info(
            ['Note that the directory must exist.',
             'Also note that this will overwrite existing files (if they exist)'])

    def _save(self, data, header, save_as_np_array=False, data2=None,
              file_guess=None, metadata=None):

        file_path = None
        self._set_save_intro(header)

        if self._plate_labels is None:
            self.set_plate_labels()

        file_path = self._get_save_file_guess(file_guess=file_guess)

        if save_as_np_array:
            try:
                np.save(file_path, data)
            except:
                return False

            if data2 is not None:

                file_path = file_path.split(".")
                file_path = ".".join(file_path[:-1]) + "2" + file_path[-1]

                try:
                    np.save(file_path, data2)
                except:
                    return False

                logging.info("Also saved secondary data as '{0}'".format(file_path))
        else:
            try:
                fs = open(file_path, 'w')
            except:
                return False

            for p in range(len(data)):
                fs.write("START PLATE {0}\n".format(self._plate_labels[p]))

                if data[p].ndim >= 2:
                    for x in xrange(data[p].shape[0]):
                        for y in xrange(data[p].shape[1]):
                            if data2 is None:
                                try:
                                    fs.write(
                                        "{0}\t{1}\t{2}\t\"{3}\"\t{4}\n".format(
                                            p, x, y,
                                            self._original_meta_data[(p, x, y)][0],
                                                "\t".join(
                                                    list(map(
                                                        str,
                                                        data[p][x, y, :])))))
                                except KeyError:

                                    logging.warning(
                                        "Position {0} does not exist".format((
                                            p, x, y)))
                                except:

                                    fs.write(
                                        "{0}\t{1}\t{2}\t\"{3}\"\t{4}\n".format(
                                            p, x, y,
                                            self._original_meta_data[(p, x, y)][0],
                                            data[p][x, y]))

                            else:
                                fs.write(
                                    "{0}\t{1}\t{2}\t\"{3}\"\t{4}\t{5}\n".format(
                                        p, x, y,
                                        self._original_meta_data[(p, x, y)][0],
                                        "\t".join(list(map(str, data[p][x, y]))),
                                        "\t".join(list(map(str, data2[p][x, y])))))

                fs.write("STOP PLATE {0}\n".format(self._plate_labels[p]))

            fs.close()

        return True

    def run(self):

        get_interactive_header("This script will take care of positional effects")
        get_interactive_info(
            ["It is in alpha-state,",
             "so let Martin know what doesn't work.",
             "And don't expect everything to work.", "",
             "Also, don't be affraid -- nothing will happen to the files you've loaded unless you decide to save over them"])

        answer = None

        while answer != 'T':

            self.print_menu()

            answer = self.get_answer()

            if answer is not None:
                self.do_task(answer)

                if answer == 'T':

                    if str(raw_input("Are you sure you wish to quit (y/N)? ")).upper() != 'Y':
                        answer = ''

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    interactive_menu = Interactive_Menu()
    if len(sys.argv) == 2:
        interactive_menu.load_file(sys.argv[1])

    interactive_menu.run()

#
# NOT IN USE
#


def vector_orth_dist(x, y, p1):
    """
    @param x: is a vector of X-measures
    @param y: is a vector of corresponding Y-measures
    @param p1: is a polynomial returned from numpy.poly1d
    """
    #
    #get the point where ax + b = 0
    x_off = -p1[0] / p1[1]
    #
    #p's unit vector creation
    p_unit = np.asarray((1 - x_off, p1(1)))
    p_unit = p_unit / np.sqrt(np.sum(p_unit ** 2))
    #
    #its orthogonal vector
    p_u_orth = np.asarray((-p_unit[1], p_unit[0]))
    #
    #distances:
    dists = np.zeros(x.shape)
    for d in xrange(dists.size):
        dists[d] = np.sum(np.asarray((x[d] - x_off, y[d])) * p_u_orth)
    #
    return dists

"""
#PLOT THE POSITIONAL EFFECT
plt.clf()
fig = plt.figure()
Ps = []
Ns = []

for p in xrange(Y2.shape[2]):
    C = c2(Y2[:,:,p], kernel, mode='nearest').ravel() / np.sum(kernel)
    z1  = np.polyfit(C, Y2[:,:,p].ravel(),1)
    Ps.append(z1)
    p1 = np.poly1d(z1)
    l = np.linspace(C.min(), C.max(), num=100)
    Ns.append( vector_orth_dist(C, Y2[:,:,p].ravel(),p1) + np.mean(Y2[:,:,p]))
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(C, Y2[:,:,p].ravel(), 'b.')
    plt.plot(l, p1(l), 'r-')
    plt.gca().annotate(s=str("%.3f" % z1[0]) + "x + " + str("%.3f" % z1[1]),
        xy=(l[5]+0.6,p1(l[5])+0.3))
    plt.xlim(1.4, 4)
    plt.ylim(1, 4.5)
    plt.ylabel('Colony Generation Time')
    plt.xlabel('How cool your neighbours are (Very <-> Not)')

fig.show()

#PLOT SECONDARY POSITIONAL EFFECTS
fig = plt.figure()
Ns2 = []

for p in xrange(Y2.shape[2]):
    C = c2(np.reshape(Ns[p], (16,24)), kernel, mode='nearest').ravel() / np.sum(kernel)
    z1  = np.polyfit(C, Ns[p],1)
    p1 = np.poly1d(z1)
    l = np.linspace(C.min(), C.max(), num=100)
    Ns2.append( vector_orth_dist(C, Ns[p], p1) + np.mean(Ns[p]))
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(C, Ns[p], 'b.')
    plt.plot(l, p1(l), 'r-')
    plt.gca().annotate(s=str("%.3f" % z1[0]) + "x + " + str("%.3f" % z1[1]),
        xy=(l[5],p1(l[5])+0.6))
    plt.xlim(2.0, 3.5)
    plt.ylim(1.5, 4.0)
    plt.ylabel('Colony Generation Time')
    plt.xlabel('How cool your neighbours are (Very <-> Not)')

plt.show()

#PLOT THE PROCESSED VS RAW HEAT MAPS
plt.clf()
fig = plt.figure()

for i in xrange(N.shape[2]):
    fig.add_subplot(2,2,i+1, title="Plate %d" % i)
    plt.boxplot([Y2[:,:,i].ravel(),
        Ns[i]])
    plt.annotate(s="std: %.3f" % np.std(Y2[:,:,i].ravel()), xy=(1,4) )
    plt.annotate(s="std: %.3f" % np.std(Ns[i]), xy=(2,4) )
    plt.ylabel('Generation Time')
    plt.xlabel('Untreated vs Normalized')
    plt.ylim(1.0,4.5)

fig.show()

#UNUSED
fig = plt.figure()

for p in xrange(Y2.shape[2]):
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(c2(Y3[:,:,p], kernel, mode='constant', cval=np.mean(Y3[:,:,p]))\
        .ravel(), Y2[:,:,p].ravel(), 'b.')
    plt.ylabel('Colony rate')
    plt.xlabel('How cool your neighbours are (Not <-> Very)')

fig.show()

#PLOT START EFFECT
fig = plt.figure()

for p in xrange(Y2.shape[2]):
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(Z2[:,:,p].ravel(), Ns[p], 'b.')
    plt.ylabel('Colony rate')
    plt.xlabel('Start-value')
    plt.xlim(0,200000)
    plt.ylim(0,4)

plt.show()

#
#
#
fs = open('slow_fast_test.csv', 'w')
y_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
yeasts = ['wt','Hog1']
fig = plt.figure()
for p in xrange(4):
    fs.write("\nPlate " + str(p+1) + "\n")
    fs.write("Row\tCol\tType\n")
    p_layout = np.random.random((8,12))>0.8
    for x in xrange(p_layout.shape[0]):
        for y in xrange(p_layout.shape[1]):
            fs.write(y_labels[x] + "\t" + str(y+1) + "\t" + yeasts[p_layout[x,y]] + "\n")
    fig.add_subplot(2,2,p+1)
    plt.imshow(p_layout, aspect='equal', interpolation='nearest')
    plt.xticks(np.arange(12), np.arange(12)+1)
    plt.yticks(np.arange(8), y_labels)
    plt.ylabel('Columns')
    plt.xlabel('Rows')

fs.close()
plt.show()
"""
