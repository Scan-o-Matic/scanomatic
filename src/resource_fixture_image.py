#!/usr/bin/env python
"""Deals with fixture analysis of an image"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import os
import time
import types
import itertools
import numpy as np
from matplotlib.pyplot import imread

#
# SCANNOMATIC LIBRARIES
#

import resource_image as resource_image
import resource_path as resource_path
import resource_config as conf
import resource_app_config as resource_app_config
import resource_logger as resource_logger


#
# DECORATORS
#


def GH_loaded_decorator(f):
    """Intended to work with Gridding History class"""
    def wrap(*args, **kwargs):
        self = args[0]
        if self._settings is None:
            if self._load() == False:
                return None
        else:
            self._settings.reload()

        return f(*args, **kwargs)

    return wrap

#
# CLASSES
#


class Gridding_History(object):
    """This class keeps track of the gridding-histories of the fixture
    using the configuration-file in the fixtures-directory"""

    plate_pinning_pattern = "plate_{0}_pinning_{1}"
    pinning_formats = ((8, 12), (16, 24), (32, 48), (64, 96))
    plate_area_pattern = "plate_{0}_area"

    def __init__(self, parent, fixture_name, paths, logger=None, app_config=None):

        if logger is None:
            logger = resource_logger.Log_Garbage_Collector()

        self._parent = parent
        self._logger = logger
        self._name = fixture_name
        self._paths = paths
        self._app_config = app_config

        self._settings = None
        self._compatibility_check()

    def __getitem__(self, key):
        #This is purposly insecure function

        if self._settings is None:
            return None

        return self._settings.get(key)

    def _get_plate_pinning_str(self, plate, pinning_format):

        return self.plate_pinning_pattern.format(plate, pinning_format)

    def _get_gridding_history(self, plate, pinning_format):

        self._settings.reload()
        return self._settings[self._get_plate_pinning_str(plate, pinning_format)]

    def _load(self):

        conf_file = conf.Config_File(self._paths.get_fixture_path(self._name))
        if conf_file.get_loaded() == False:
            self._settings = None
            return False
        else:
            self._settings = conf_file
            return True

    @GH_loaded_decorator
    def get_gridding_history(self, plate, pinning_format):

        h = self._get_gridding_history(plate, pin_format)

        if h is None:
            return None

        return np.array(h.values())

    @GH_loaded_decorator
    def set_gridding_parameters(self, project_id, pinning_format, plate,
            center, spacings):

        h = self._get_gridding_history(plate, pinning_format)

        if h is None:

            h = {}

        h[project_id] = center + spacings

        f = self._settings
        f.set(self._get_plate_pinning_str(plate, pinning_format), h)
        f.save()

    @GH_loaded_decorator
    def unset_gridding_parameters(self, project_id, pinning_format, plate):

        h = self._get_gridding_history(plate, pinning_format)

        if h is None:

            return None

        try:
            del h[project_id]
        except:
            self._logger.warning(("Gridding history for {0} project {1}"
                " plate {2} pinning format {3} did not exist, thus"
                " nothing to delete").format(self._name,
                project_id, plate, pinning_format))

        f = self._settings
        f.set(self._get_plate_pinning_str(plate, pinning_format), h)
        f.save()
        
    @GH_loaded_decorator
    def reset_gridding_history(self, plate):

        f = self._settings

        for pin_format in self.pinning_formats:
            f.set(self._get_plate_pinning_str(plate, pin_format), {})

        f.save()

    @GH_loaded_decorator
    def reset_all_gridding_histories(self):

        f = self._settings
        plate = True
        i = 0

        while plate is not None:

            plate = f.get(self.plate_area_pattern.format(i))
            if plate is not None:

                self.reset_gridding_history(i)

            i += 1

    @GH_loaded_decorator
    def _compatibility_check(self):
        '''As of version 0.998 a change was made to the structure of pinning
        history storing, both what is stored and how. Thus older histories must
        be cleared. When done, version is changed to 0.998'''

        f = self._settings
        v = f.get('version')
        if (v < self._app_config.version_fixture_grid_history_change_1):

            self.reset_all_gridding_histories()
            f.set('version', self._app_config.version_fixture_grid_history_change_1)
            f.save()
        
class Fixture_Image(object):

    def __init__(self, fixture, image_path=None,
            image=None, markings=None, define_reference=False,
            fixture_directory=None, markings_path=None,
            im_scale=None, logger=None):

        if logger is None:
            logger = resource_logger.Log_Garbage_Collector()

        self._logger = logger

        self._paths = resource_path.Paths()
        self._config = resource_app_config.Config(self._paths)

        self._history = Gridding_History(self, fixture, self._paths, 
            logger=logger, app_config = self._config)


        self._define_reference = define_reference
        self.fixture_name = fixture
        self.im_scale = im_scale
        self.im_original_scale = (im_scale is None and 1.0 or im_scale)

        self.set_reference_file(fixture,
            fixture_directory=fixture_directory,
            image_path=image_path)

        self.set_marking_path(markings_path)

        self.set_number_of_markings(markings)

        self._markers_X = None
        self._markers_Y = None
        self._gs_values = None
        self._gs_indices = None
        self.im = None

        self.set_image(image=image, image_path=image_path)

    def _output_f(self, *args, **kwargs):

        print "Debug output function: ", args, kwargs

    def __getitem__(self, key):

        if key in ['image']:

            return self.im

        elif key in ['current']:

            return self.fixture_current

        elif key in ['fixture']:

            return self.fixture_reference

        elif key in ['fixture-path']:

            return self._fixture_reference_path

        elif key in ['name']:

            return self.fixture_name

        elif key in  ["grayscale", "greyscale"]:

            return self._gs_indices, self._gs_values

        elif key in ["markers", "marks"]:

            return self._markers_X, self._markers_Y

        elif key in ["ref-markers"]:

            return self._get_markings(source="fixture")

        elif key in ["plates"]:

            return self.get_plates()

        elif key in ['version', 'Version']:

            return self.fixture_reference.get('version')
 
        elif key in ['history', 'pinning', 'pinnings', 'gridding']:

            return self._history

        else:

            print "***ERROR: Unknown key {0}".format(key)

    def __setitem__(self, key, val):

        if key in ['grayscale-area', 'grayscale-coords', 'gs-area',
            'gs-coords', 'greyscale-area', 'greyscale-coords']:

            self.fixture_current['grayscale_area'] = val

        elif key in ['grayscale', 'greyscale', 'gs']:

            if val is None or len(val) == 0:
                has_gs = False
            else:
                has_gs = True

            self.fixture_current['grayscale'] = has_gs
            self.fixture_current['grayscale_indices'] = val

        elif key in ['plate-coords']:

            try:
                plate, coords = val
                plate = int(plate)
            except:
                print "***ERROR: Must be a tuple/list of plate index and coords"
                return

            plate_str = "plate_{0}_area".format(plate)
            self.fixture_current[plate_str] = coords

        else:

            print "***ERROR: Failed to set {0} to {1}".format(key, val)

    def _load_reference(self):

        fixture_path = self._fixture_reference_path

        self._logger.info("FIXTURE: Reference loaded from {0}".format(
            fixture_path))

        self.fixture_reference = conf.Config_File(fixture_path)

        cur_name = self.fixture_reference.get('name')
        if cur_name is None or cur_name == "":
            self.fixture_reference.set('name', self._paths.get_fixture_name(self.fixture_name))

        if self._define_reference:
            self.fixture_current = self.fixture_reference
        else:
            self.fixture_current = conf.Config_File(fixture_path + "_tmp")

    def set_number_of_markings(self, markings):

        self.markings = None
        if markings is not None:

            try:
                self.markings = int(markings)
            except:
                pass

        if self.markings is None:

            try:

                self.markings = int(self.fixture_reference.get("marker_count"))
            except:
                self.markings = 3

        if self._define_reference:

            self['fixture'].set("marker_count", self.markings)

    def set_marking_path(self, marking_path):

        if marking_path is not None:

            self.marking_path = marking_path

        else:

            self.marking_path = self.fixture_reference.get("marker_path")

        if self._define_reference:

            self['fixture'].set("marker_path", self.marking_path)

    def set_image(self, image=None, image_path=None):

        if image is not None:

            self.im = image

        elif image_path is not None:

            self.im = imread(image_path)

        else:

            self.im = None

    def set_reference_file(self, fixture_name, fixture_directory=None,
            image_path=None):

        if fixture_directory is not None:

            self._fixture_reference_path = \
                self._paths.get_fixture_path(fixture_name,
                own_path=fixture_directory)

            self._fixture_config_root = fixture_directory

        elif image_path is not None:

            self._fixture_config_root = \
                os.sep.join(image_path.split(os.sep)[:-1])

            self._fixture_reference_path = \
                self._paths.get_fixture_path(fixture_name,
                own_path=self._fixture_config_root)

        else:

            self._fixture_config_root = "."
            self._fixture_reference_path = self._paths.get_fixture_path(
                fixture_name, own_path="")

        self._load_reference()

    def threaded(self, output_function=None):

        if output_function is None:
            output_function = self._output_f

        t = time.time()
        output_function('Fixture calibration',
                    "Threading invokes marker analysis", "LA",
                    debug_level='info')

        self.run_marker_analysis()

        output_function('Fixture calibration',
                    "Threading marker detection complete, invokes setting area positions" +
                    " (acc-time {0} s)".format(time.time()-t), "LA",
                    debug_level='info')

        self.set_current_areas()

        output_function('Fixture calibration',
                    "Threading areas set(acc-time: {0} s)".format(time.time()-t),
                    "LA", debug_level='info')

        self.analyse_grayscale()

        output_function('Fixture calibration',
                    "Threading done (took: {0} s)".format(time.time()-t),
                    "LA", debug_level='info')



    def _get_markings(self, source='fixture'):

        X = []
        Y = []

        if self.markings == 0 or self.markings is None:

            return None, None

        for m in xrange(self.markings):

            Z = self[source]['marking_{0}'.format(m)]

            if Z is not None:

                X.append(Z[0])
                Y.append(Z[1])


        if len(X) == 0:

            return None, None

        return np.array(X), np.array(Y)

    def run_marker_analysis(self, output_function=None):

        if output_function is None:

            output_function = self._output_f

        t = time.time()

        if self.marking_path == None or self.markings < 1:

            msg = "Error, no marker set ('%s') or no markings (%s)." % (
                self.marking_path, self.markings)

            output_function('Fixture calibration: Marker Detection', msg, "LA",
                        debug_level='error')

            return None

        if self._define_reference:

            analysis_im_path = \
                self._paths.fixture_image_file_pattern.format(
                self.fixture_name)

            #analysis_im_path = self._fixture_config_root + os.sep + \
            #        self.fixture_name + ".tiff"

            target_conf_file = self.fixture_reference

        else:

            analysis_im_path = None
            target_conf_file = self.fixture_current

        target_conf_file.set("version", __version__)

        output_function('Fixture calibration: Marker Detection', "Scaling image", "LA",
                        debug_level='info')

        if self.im_scale is not None:

            analysis_img = self.im
            scale_str = "Kept scale {0}".format(self.im_scale)
        else:

            analysis_img = resource_image.Quick_Scale_To_im(im=self.im, scale=0.25)
            scale_str = "New scale {0}".format(0.25)
            self.im_scale = 0.25

        output_function('Fixture calibration: Marker Detection', scale_str, "LA",
                        debug_level='info')

        output_function('Fixture calibration: Marker Detection', "Scaled (acc {0} s)".format(
                        time.time()-t), "LA",
                        debug_level='info')

        if analysis_im_path is not None:

            #from matplotlib.image import imsave
            #imsave(analysis_im_path, analysis_img, format='tiff')
            np.save(analysis_im_path, analysis_img)

        msg = "Setting up Image Analysis (acc {0} s)".format(time.time()-t)

        output_function('Fixture calibration: Marker Detection', msg, 'A',
                    debug_level='debug')

        im_analysis = resource_image.Image_Analysis(
                    image=analysis_img,
                    pattern_image_path=self.marking_path,
                    scale=self.im_scale)

        msg = "Finding pattern (acc {0} s)".format(time.time()-t)

        output_function('Fixture calibration, Marker Detection', msg, 'A',
                    debug_level='debug')

        Xs, Ys = im_analysis.find_pattern(markings=self.markings)

        self._markers_X = Xs
        self._markers_Y = Ys

        if Xs is None or Ys is None:

            output_function('Fixture error', "No markers found")

        elif len(Xs) == self.markings:

            self._set_markings_in_conf(target_conf_file, Xs, Ys)
            output_function("Fixture calibration", "Setting makers {0}, {1}".format(
                Xs, Ys))

        msg = "Marker Detection complete (acc {0} s)".format(time.time()-t)
        output_function('Fixture calibration: Marker Detection', msg, 'A',
                    debug_level='debug')

        return analysis_im_path

    def _set_markings_in_conf(self, conf_file, Xs, Ys):

        if Xs is not None:

            for i in xrange(len(Xs)):
                conf_file.set("marking_" + str(i), (Xs[i], Ys[i]))

            conf_file.set("marking_center_of_mass", (Xs.mean(), Ys.mean()))

    def _version_check_positions_arr(self, *args):
        """Note that it only works for NP-ARRAYS and NOT for lists"""
        version = self['fixture']['version']
        if version is None or \
            version < self._config.version_first_pass_change_1:

            args = list(args)
            for a in args:
                if a is not None:
                    a *= 4

    def _get_markings_rotations(self):

        #CURRENT SETTINGS
        X, Y = self._get_markings(source="current")

        if X is None or Y is None:
            return None, None

        X = np.array(X)
        Y = np.array(Y)
        Mcom = self['current']['marking_center_of_mass']
        if Mcom is None:
            Mcom = np.array((X.mean(), Y.mean()))
        else:
            Mcom = np.array(Mcom)
        dX = X - Mcom[0]
        dY = Y - Mcom[1]

        L = np.sqrt(dX ** 2 + dY ** 2)

        #FIXTURE SETTINGS
        version = self['fixture']['version']
        refX, refY = self._get_markings(source="fixture")
        refX = np.array(refX)
        refY = np.array(refY)
        ref_Mcom = np.array(self['fixture']["marking_center_of_mass"])
        self._version_check_positions_arr(ref_Mcom, refX, refY)
        ref_dX = refX - ref_Mcom[0]
        ref_dY = refY - ref_Mcom[1]

        ref_L = np.sqrt(ref_dX ** 2 + ref_dY ** 2)

        if Y.shape == refX.shape == refY.shape:

            #Find min diff order
            s_reseed = range(len(ref_L))
            s = range(len(L))

            tmp_dL = []
            tmp_s = []
            for i in itertools.permutations(s):

                tmp_dL.append((L[list(i)] - ref_L) ** 2)
                tmp_s.append(i)

            dLs = np.array(tmp_dL).sum(1)
            s = list(tmp_s[dLs.argmin()])

            print "** Found sort order that matches the reference", s,
            print ". Error:", np.sqrt(dLs.min())
            #Quality control of all the markers so that none is bad
            #Later

            #Rotations
            A = np.arccos(dX / L)
            A = A * (dY > 0) + -1 * A * (dY < 0)

            ref_A = np.arccos(ref_dX / ref_L)
            ref_A = ref_A * (ref_dY > 0) + -1 * ref_A * (ref_dY < 0)

            dA = A[s] - ref_A

            d_alpha = dA.mean()
            print "** Found average rotation", d_alpha,
            print "from set of delta_rotations:", dA

            #Setting the current marker order so it matches the
            #Reference one according to the returns variables!
            self._set_markings_in_conf(self['current'], X[s], Y[s])

            return d_alpha, Mcom

        else:

            print "*** ERROR: Missmatch in number of markings"
            return None

    def get_subsection(self, section, scale=1.0):

        im = self['image']

        if im is not None and section is not None:

            section = zip(*map(sorted, zip(*section)))

            try:

                subsection = im[
                    section[0][1] * scale: section[1][1] * scale,
                    section[0][0] * scale: section[1][0] * scale]

            except:

                subsection = None

            return subsection

        return None


    def analyse_grayscale(self):

        im = self.get_subsection(self['current']['grayscale_area'],
                scale=self.im_original_scale)

        print id(im), type(im), "For save", self['current']['grayscale_area'], self.im_original_scale

        if im is None or 0 in im.shape:
            return False

        np.save(".tmp.npy", im)
        ag = resource_image.Analyse_Grayscale(target_type="Kodak", 
            image=im, scale_factor=self.im_original_scale)
        
        gs_indices = ag.get_target_values()
        gs_values = ag.get_source_values()
        self._gs_values = gs_values
        self._gs_indices = gs_indices

        if self._define_reference:

            self['fixture'].set('grayscale_indices', gs_indices)

    def _get_rotated_point(self, point, alpha, offset=(0, 0)):

        if alpha is None:
            return (None, None)

        tmp_l = np.sqrt(point[0] ** 2 + point[1] ** 2)
        tmp_alpha = np.arccos(point[0] / tmp_l)

        tmp_alpha = tmp_alpha * (point[1] > 0) + -1 * tmp_alpha * \
                            (point[1] < 0)

        new_alpha = tmp_alpha + alpha
        new_x = np.cos(new_alpha) * tmp_l + offset[0]
        new_y = np.sin(new_alpha) * tmp_l + offset[1]

        return (new_x, new_y)

    def set_current_areas(self):

        alpha, Mcom = self._get_markings_rotations()
        X, Y = self._get_markings(source='current') 
        ref_Mcom = np.array(self['fixture']["marking_center_of_mass"])

        self['current'].flush()
        self._set_markings_in_conf(self['current'], X, Y)

        ref_gs = self['fixture']["grayscale_area"]
        version = self['fixture']['version']

        self._version_check_positions_arr(ref_Mcom)

        if version is None or \
            version < self._config.version_first_pass_change_1:

            scale_factor = 4
        else:
            scale_factor = 1
 
        if ref_gs is not None and bool(self['fixture']["grayscale"]) == True:

            dGs1 = scale_factor * np.array(ref_gs[0]) - ref_Mcom
            dGs2 = scale_factor * np.array(ref_gs[1]) - ref_Mcom

            self['current'].set("grayscale_area",
                [self._get_rotated_point(dGs1, alpha, offset=Mcom),
                self._get_rotated_point(dGs2, alpha, offset=Mcom)])

        i = 0
        ref_m = True
        p_str = "plate_{0}_area"
        f_plates = self['fixture'].get_all("plate_%n_area")

        for i, p in enumerate(f_plates):

            dM1 = scale_factor * np.array(p[0]) - ref_Mcom
            dM2 = scale_factor * np.array(p[1]) - ref_Mcom

            self['current'].set(p_str.format(i),
                [self._get_rotated_point(dM1, alpha, offset=Mcom),
                self._get_rotated_point(dM2, alpha, offset=Mcom)])

    def _get_coords_sorted(self, coords):

        return zip(*map(sorted, zip(*coords)))

    def get_plates(self, source="current", indices=False):

        plate_list = []

        p = True
        ps = "plate_{0}_area"
        i = 0

        while p is not None:

            p = self[source][ps.format(i)]

            if p is not None:

                if indices:
                    plate_list.append(i)
                else:
                    p = self._get_coords_sorted(p)
                    plate_list.append(p)

            i += 1

        return plate_list
