"""Deals image as used by first pass analysis"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import os
import time
import itertools
import numpy as np
from matplotlib.pyplot import imread
#import weakref

#
# SCANNOMATIC LIBRARIES
#

import scanomatic.io.paths as paths
import scanomatic.io.config_file as config_file
import scanomatic.io.app_config as app_config

import scanomatic.io.logger as logger

import imageBasics
import imageFixture
import imageGrayscale
import grayscale

#
# DECORATORS
#


def GH_loaded_decorator(f):
    """Intended to work with Gridding History class"""
    def wrap(*args, **kwargs):
        self = args[0]
        if self._settings is None:
            if self._load() is False:
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

    def __init__(self, parent, fixture_name, paths, app_config=None):

        #self._parent = weakref.ref(parent) if parent else None

        self._logger = logger.Logger("Gridding History")
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

        #self._settings.reload()
        return self._settings[self._get_plate_pinning_str(plate, pinning_format)]

    def _load(self):

        conf_file = config_file.Config_File(self._paths.get_fixture_path(self._name))
        if conf_file.get_loaded() is False:
            self._settings = None
            return False
        else:
            self._settings = conf_file
            return True

    @GH_loaded_decorator
    def get_gridding_history(self, plate, pinning_format):

        h = self._get_gridding_history(plate, pinning_format)

        if h is None:
            self._logger.info(
                "No history in {2} on plate {0} format {1}".format(
                    plate, pinning_format, self._name))
            return None

        self._logger.info(
            "Returning history for {0} plate {1} format {2}".format(
            self._name, plate, pinning_format))
        return np.array(h.values())

    @GH_loaded_decorator
    def get_gridding_history_specific_plate(self, p_uuid, plate,
                                            pinning_format):

        h = self._get_gridding_history(plate, pinning_format)
        if h is None or p_uuid not in h:
            self._logger.info(
                "No history in {2} on plate {0} format {1}".format(
                    plate, pinning_format, self._name))
            return None

        self._logger.info(
            "Returning history for {0} plate {1} format {2} uuid {3}".format(
            self._name, plate, pinning_format, p_uuid))

        return h[p_uuid]

    @GH_loaded_decorator
    def set_gridding_parameters(self, project_id, pinning_format, plate,
                                center, spacings):

        h = self._get_gridding_history(plate, pinning_format)

        if h is None:

            h = {}

        h[project_id] = center + spacings

        self._logger.info("Setting history {0} on {1} for {2} {3}".format(
            center + spacings, self._name, project_id, plate))

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
            self._logger.warning((
                "Gridding history for {0} project {1}"
                " plate {2} pinning format {3} did not exist, thus"
                " nothing to delete").format(
                    self._name, project_id, plate, pinning_format))
            return False

        f = self._settings
        f.set(self._get_plate_pinning_str(plate, pinning_format), h)
        f.save()

        return True

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


class Image(object):

    MARKER_DETECTION_SCALE = 0.25
    EXPECTED_IM_SIZE = (6000, 4800)

    def __init__(self, fixture, image_path=None,
                 image=None, markings=None, define_reference=False,
                 fixture_directory=None, markings_path=None,
                 im_scale=None, appConfig=None):

        self._logger = logger.Logger("Fixture Image")

        self._paths = paths.Paths()

        if appConfig is None:
            self._config = app_config.Config(self._paths)
        else:
            self._config = appConfig

        self._define_reference = define_reference
        self.fixture_name = fixture
        self.im_scale = im_scale
        self.im_original_scale = im_scale

        self.set_reference_file(
            fixture,
            fixture_directory=fixture_directory,
            image_path=image_path)

        f_name = self.get_name_in_ref()
        if f_name is None:
            f_name = fixture
        self._history = Gridding_History(
            self, f_name, self._paths,
            app_config=self._config)

        self._markers_X = None
        self._markers_Y = None
        self._gs_values = None
        self._gs_indices = None
        self.im = None

        self.set_image(image=image, image_path=image_path)
        self.set_marking_path(markings_path)
        self.set_number_of_markings(markings)

        #print fixture, self['fixture-path']

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

        elif key in ["grayscaleTarget"]:

            refTarget = self.fixture_reference.get("grayscale_indices")

            if refTarget is not None:
                return refTarget
            elif self._gs_indices is not None:
                return self._gs_indices
            else:
                return grayscale.getGrayscaleTargets(self['grayscale_type'])

        elif key in ["grayscaleSource"]:

            return self._gs_values

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

        elif key in ['scale']:

            return self.im_original_scale

        elif key in ['grayscale_type', 'grayscaleType', 'grayscaleName']:

            gs_type = self.fixture_reference.get('grayscale_type')
            if gs_type is None:
                self._logger.warning("Using default Grayscale")
                gs_type = grayscale.getDefualtGrayscale()
            return gs_type
        else:

            raise Exception("***ERROR: Unknown key {0}".format(key))

    def __setitem__(self, key, val):

        if key in ['grayscale-area', 'grayscale-coords', 'gs-area',
                   'gs-coords', 'greyscale-area', 'greyscale-coords']:

            self.fixture_current['grayscale_area'] = val

        elif key in ['grayscale_type']:

            if val in grayscale.getGrayscales():
                self.fixture_current['grayscale_type'] = val
            else:
                self.fixture_current['grayscale_type'] = None

            self._setCurrentFixtureHasGrayscale()

        elif key in ['grayscale', 'greyscale', 'gs']:

            self.fixture_current['grayscale_indices'] = val
            self._setCurrentFixtureHasGrayscale()

        elif key in ['plate-coords']:

            try:
                plate, coords = val
                plate = int(plate)
            except:
                self._logger.error(
                    "Plate coordinates must be a tuple/list of " +
                    "plate index and coords")
                return

            plate_str = "plate_{0}_area".format(plate)
            self.fixture_current[plate_str] = coords

        else:

            raise("Failed to set {0} to {1}, key unkown".format(key, str(val)))

    def _setCurrentFixtureHasGrayscale(self):

        indices = self.fixture_current['grayscale']
        if ((indices is None or isinstance(indices, bool) or
                len(indices) == 0) and
                self.fixture_current['grayscale_type'] in (None, '')):
            has_gs = False
        else:
            has_gs = True

        self._logger.info("The grayscale status is set to {0}".format(has_gs))

        self.fixture_current['grayscale'] = has_gs

    def _load_reference(self):

        fixture_path = self._fixture_reference_path

        self._logger.info("Reference fixture loaded from {0}".format(
            fixture_path))

        self.fixture_reference = config_file.Config_File(fixture_path)

        cur_name = self.fixture_reference.get('name')
        if cur_name is None or cur_name == "":
            self.fixture_reference.set('name', self._paths.get_fixture_name(self.fixture_name))

        if self._define_reference:
            self.fixture_current = self.fixture_reference
        else:
            self.fixture_current = config_file.Config_File(fixture_path + "_tmp")

    def get_name_in_ref(self):

        name_in_file = self.fixture_reference.get('name')
        name_from_file = self._paths.get_fixture_name(
            self._fixture_reference_path)

        if name_in_file != name_from_file:
            self._logger.warning(
                "Missmatch in fixture name in file compared to file name! " +
                "In used file: '{0}', In system reference: '{1}'".format(
                    name_in_file, name_from_file))

        return name_from_file

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

        if marking_path is not None and os.path.isfile(marking_path):

            self.marking_path = marking_path

        else:

            self.marking_path = self._paths.marker

        """
        else:

            self.marking_path = self.fixture_reference.get("marker_path")

        self._logger.info("Marker set to: {0} {1}".format(self.marking_path, type(self.marking_path)))
        """

        if self._define_reference:

            self['fixture'].set("marker_path", self.marking_path)

    def set_im_scale(self):

        if self.im is not None and self.im_original_scale is None:

            #Check so equally scaled both dimensions
            scale_d1, scale_d2 = [
                self.im.shape[i] / float(self.EXPECTED_IM_SIZE[i])
                for i in range(2)]

            if abs(scale_d1 - scale_d2) < 0.01:

                self.im_original_scale = (scale_d1 + scale_d2) / 2.0

    def set_image(self, image=None, image_path=None):

        if image is not None:

            self.im = image

        elif image_path is not None:

            self.im = imread(image_path)
            if self.im is None:
                self._logger.error("Could not load image")

            else:
                self._logger.info("Loaded image {0} with shape {1}".format(
                    image_path, self.im.shape))
        else:

            self._logger.warning(
                "No information supplied about how to load image," +
                "thuse none loaded")
            self.im = None

        self.set_im_scale()

    def set_reference_file(self, fixture_name, fixture_directory=None,
                           image_path=None):

        if fixture_directory is not None:

            self._fixture_reference_path = \
                self._paths.get_fixture_path(
                    fixture_name,
                    own_path=fixture_directory)

            self._logger.info(
                "Refernce set to " +
                "{0} by using fixture directory: {1} and {2}".format(
                    self._fixture_reference_path,
                    fixture_directory,
                    fixture_name))

            self._fixture_config_root = fixture_directory

        elif image_path is not None:

            self._fixture_config_root = os.path.dirname(image_path)

            self._fixture_reference_path = self._paths.get_fixture_path(
                fixture_name,
                own_path=self._fixture_config_root)

            self._logger.info(
                "Refernce set to {0} by using image path: {1} and {2}".format(
                    self._fixture_reference_path,
                    image_path,
                    fixture_name))

        else:

            self._fixture_config_root = "."
            self._fixture_reference_path = self._paths.get_fixture_path(
                fixture_name, own_path="")

            self._logger.info(
                "Reference set to " +
                "{0} by using current directory and name {1}".format(
                    self._fixture_reference_path,
                    fixture_name))

        self._load_reference()

    def threaded(self):

        logger = self._logger
        t = time.time()
        logger.debug("Threading invokes marker analysis")

        self.run_marker_analysis()

        logger.debug(
            "Threading marker detection complete," +
            "invokes setting area positions (acc-time {0} s)".format(
                time.time() - t))

        self.set_current_areas()

        logger.debug(
            "Threading areas set(acc-time: {0} s)".format(time.time() - t))

        self.analyse_grayscale()

        logger.debug(
            "Grayscale ({0}) analysed (acc-time: {1} s)".format(
                self['grayscale_type'],
                time.time() - t))

        logger.debug(
            "Threading done (took: {0} s)".format(time.time() - t))

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

    def run_marker_analysis(self):

        logger = self._logger

        t = time.time()

        if self.marking_path is None or self.markings < 1:

            logger.error(
                "No marker set ('{0}') or no markings ({1}).".format(
                    self.marking_path, self.markings))

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

        logger.debug("Scaling image")

        if self.im_scale is not None:

            analysis_img = self.im
        else:

            analysis_img = imageBasics.Quick_Scale_To_im(
                im=self.im,
                scale=self.MARKER_DETECTION_SCALE / self.im_original_scale)

            self.im_scale = self.MARKER_DETECTION_SCALE / self.im_original_scale

        logger.debug(
            "New scale {0} (acc {1} s)".format(
                self.MARKER_DETECTION_SCALE / self.im_original_scale,
                time.time() - t))

        if analysis_im_path is not None:

            #from matplotlib.image import imsave
            #imsave(analysis_im_path, analysis_img, format='tiff')
            np.save(analysis_im_path, analysis_img)

        logger.debug(
            "Setting up Image Analysis (acc {0} s)".format(time.time() - t))

        im_analysis = imageFixture.FixtureImage(
            image=analysis_img,
            pattern_image_path=self.marking_path,
            scale=self.im_scale,
            resource_paths=self._paths)

        logger.debug(
            "Finding pattern (acc {0} s)".format(time.time() - t))

        Xs, Ys = im_analysis.find_pattern(markings=self.markings)

        self._markers_X = Xs
        self._markers_Y = Ys

        if Xs is None or Ys is None:

            logger.error("No markers found")

        elif len(Xs) == self.markings:

            self._set_markings_in_conf(target_conf_file, Xs, Ys)
            logger.debug("Setting makers {0}, {1}".format(
                Xs, Ys))

        logger.debug(
            "Marker Detection complete (acc {0} s)".format(time.time() - t))

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
        #print "Rotation in", X, Y, Mcom
        if Mcom is None:
            Mcom = np.array((X.mean(), Y.mean()))
        else:
            Mcom = np.array(Mcom)
        dX = X - Mcom[0]
        dY = Y - Mcom[1]

        L = np.sqrt(dX ** 2 + dY ** 2)

        #FIXTURE SETTINGS
        #version = self['fixture']['version']
        refX, refY = self._get_markings(source="fixture")
        refX = np.array(refX) * self.im_original_scale
        refY = np.array(refY) * self.im_original_scale
        ref_Mcom = np.array(self['fixture']["marking_center_of_mass"])
        ref_Mcom *= self.im_original_scale
        self._version_check_positions_arr(ref_Mcom, refX, refY)
        ref_dX = refX - ref_Mcom[0]
        ref_dY = refY - ref_Mcom[1]

        ref_L = np.sqrt(ref_dX ** 2 + ref_dY ** 2)

        if Y.shape == refX.shape == refY.shape:

            #Find min diff order
            #s_reseed = range(len(ref_L))
            s = range(len(L))

            tmp_dL = []
            tmp_s = []
            for i in itertools.permutations(s):

                tmp_dL.append((L[list(i)] - ref_L) ** 2)
                tmp_s.append(i)

            dLs = np.array(tmp_dL).sum(1)
            s = list(tmp_s[dLs.argmin()])

            self._logger.debug(
                "Found sort order that matches the reference" +
                "{0} (error {1})".format(s, np.sqrt(dLs.min())))
            #Quality control of all the markers so that none is bad
            #Later

            #Rotations
            A = np.arccos(dX / L)
            A = A * (dY > 0) + -1 * A * (dY < 0)

            ref_A = np.arccos(ref_dX / ref_L)
            ref_A = ref_A * (ref_dY > 0) + -1 * ref_A * (ref_dY < 0)

            dA = A[s] - ref_A

            d_alpha = dA.mean()
            self._logger.debug("Found average rotation {0} from {1}".format(
                d_alpha, dA))

            #Setting the current marker order so it matches the
            #Reference one according to the returns variables!
            self._set_markings_in_conf(self['current'], X[s], Y[s])

            return d_alpha, Mcom

        else:

            self._logger.critical("Missmatch in number of markings!")
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
                                 scale=1.0)

        """
        print id(im), type(im), "For save", self['current']['grayscale_area'], self.im_original_scale
        """

        if im is None or 0 in im.shape:
            self._logger.error(
                "No valid grayscale area (Current area: {0})".format(
                    self['current']['grayscale_area']))
            return False

        #np.save(".tmp.npy", im)
        ag = imageGrayscale.Analyse_Grayscale(
            target_type=self['grayscale_type'], image=im,
            scale_factor=self.im_original_scale)

        gs_indices = ag.get_target_values()
        self._gs_indices = gs_indices
        gs_values = ag.get_source_values()
        self._gs_values = gs_values

        """Outdated since default is to just use fixture name
        if self._define_reference:

            self['fixture'].set('grayscale_indices', gs_indices)
        """

    def _get_rotated_point(self, point, alpha, offset=(0, 0)):
        """Returns a rotated and offset point.

        Parameters
        ==========

        point : array-like
            A two position array for the source position

        alpha : float
            Rotation angle 

        offset : arrary-like, optional
            The offset of the point / how much it will be moved after the
            rotation.
            Default is to not move.
        """

        if alpha is None:
            return (None, None)

        tmp_l = np.sqrt(point[0] ** 2 + point[1] ** 2)
        tmp_alpha = np.arccos(point[0] / tmp_l)

        tmp_alpha = (tmp_alpha * (point[1] > 0) + -1 * tmp_alpha *
                     (point[1] < 0))

        new_alpha = tmp_alpha + alpha
        new_x = np.cos(new_alpha) * tmp_l + offset[0]
        new_y = np.sin(new_alpha) * tmp_l + offset[1]

        if new_x > self.EXPECTED_IM_SIZE[0]:
            self._logger.warning(
                    "Point X-value ({1}) outside image {0}".format(
                        self._name, new_x))
            new_x = self.EXPECTED_IM_SIZE[0]
        elif new_x < 0:
            self._logger.warning(
                    "Point X-value ({1}) outside image {0}".format(
                        self._name, new_x))
            new_x = 0


        if new_y > self.EXPECTED_IM_SIZE[1]:
            self._logger.warning(
                    "Point Y-value ({1}) outside image {0}".format(
                        self._name, new_y))
            new_y = self.EXPECTED_IM_SIZE[1]
        elif new_y < 0:
            self._logger.warning(
                    "Point Y-value ({1}) outside image {0}".format(
                        self._name, new_y))
            new_y = 0

        return (new_x, new_y)

    def set_current_areas(self):

        alpha, Mcom = self._get_markings_rotations()
        X, Y = self._get_markings(source='current')
        ref_Mcom = np.array(self['fixture']["marking_center_of_mass"])
        #print ref_Mcom, self.im_original_scale
        #ref_Mcom *= self.im_original_scale

        dMcom = Mcom - ref_Mcom

        #print "dMcom", dMcom, alpha

        self['current'].flush()
        self._set_markings_in_conf(self['current'], X, Y)

        version = self['fixture']['version']
        ref_gs = np.array(self['fixture']["grayscale_area"])
        ref_gs *= self.im_original_scale

        self._version_check_positions_arr(ref_Mcom)

        if (version is None or
                version < self._config.version_first_pass_change_1):

            scale_factor = 4.0

        else:

            scale_factor = 1.0

        if ref_gs is not None and bool(self['fixture']["grayscale"]) is True:

            Gs1 = scale_factor * ref_gs[0]
            Gs2 = scale_factor * ref_gs[1]

            self['current'].set("grayscale_area",
                [self._get_rotated_point(Gs1, alpha, offset=dMcom),
                 self._get_rotated_point(Gs2, alpha, offset=dMcom)])

        else:

            if bool(self['fixture']['grayscale']) is not True:

                self._logger.warning("No grayscale enabled in reference")

            if ref_gs is None:

                self._logger.warning("No grayscale area in reference")
            #print Gs1, self._get_rotated_point(Gs1, alpha, offset=dMcom)

        i = 0
        #ref_m = True
        p_str = "plate_{0}_area"
        f_plates = self['fixture'].get_all("plate_%n_area")

        for i, p in enumerate(f_plates):
            p = np.array(p)
            p *= self.im_original_scale
            M1 = scale_factor * p[0]
            M2 = scale_factor * p[1]

            self['current'].set(p_str.format(i),
                [self._get_rotated_point(M1, alpha, offset=dMcom),
                 self._get_rotated_point(M2, alpha, offset=dMcom)])

            #print M1, self._get_rotated_point(M1, alpha, offset=dMcom)

    def _set_current_areas(self):

        alpha, Mcom = self._get_markings_rotations()
        X, Y = self._get_markings(source='current')
        ref_Mcom = np.array(self['fixture']["marking_center_of_mass"])
        ref_Mcom *= self.im_original_scale

        self['current'].flush()
        self._set_markings_in_conf(self['current'], X, Y)

        version = self['fixture']['version']
        ref_gs = np.array(self['fixture']["grayscale_area"])
        ref_gs *= self.im_original_scale

        self._version_check_positions_arr(ref_Mcom)

        if (version is None or
                version < self._config.version_first_pass_change_1):

            scale_factor = 4.0

        else:

            scale_factor = 1.0

        if ref_gs is not None and bool(self['fixture']["grayscale"]) is True:

            dGs1 = scale_factor * ref_gs[0] - ref_Mcom
            dGs2 = scale_factor * ref_gs[1] - ref_Mcom

            self['current'].set("grayscale_area",
                [self._get_rotated_point(dGs1, alpha, offset=Mcom),
                 self._get_rotated_point(dGs2, alpha, offset=Mcom)])

        i = 0
        #ref_m = True
        p_str = "plate_{0}_area"
        f_plates = self['fixture'].get_all("plate_%n_area")

        for i, p in enumerate(f_plates):
            p = np.array(p)
            p *= self.im_original_scale
            dM1 = scale_factor * p[0] - ref_Mcom
            dM2 = scale_factor * p[1] - ref_Mcom

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
