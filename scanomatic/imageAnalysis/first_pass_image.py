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

import time
import itertools
import numpy as np
from matplotlib.pyplot import imread

#
# SCANNOMATIC LIBRARIES
#

from scanomatic.io.grid_history import GriddingHistory
from scanomatic.io.logger import Logger

import imageBasics
import imageFixture
import imageGrayscale


def _get_coords_sorted(coords):

    return zip(*map(sorted, zip(*coords)))


def get_image_scale(im):

    small_error = 0.01
    invalid_scale = -1.0

    if im:

        scale_d1, scale_d2 = [im.shape[i] / float(FixtureImage.EXPECTED_IM_SIZE[i]) for i in range(2)]

        if abs(scale_d1 - scale_d2) < small_error:

            return (scale_d1 + scale_d2) / 2.0

        return invalid_scale


class FixtureImage(object):

    MARKER_DETECTION_SCALE = 0.25
    EXPECTED_IM_SIZE = (6000, 4800)

    def __init__(self, fixture=None):

        """

        :type fixture: scanomatic.io.fixtures.FixtureSettings
        """
        self._logger = Logger("Fixture Image")

        self._reference_fixture_settings = fixture
        self._current_fixture_settings = None
        """:type : scanomatic.io.fixtures.Fixture_Settings"""
        self._history = GriddingHistory(fixture)

        self.im = None
        self.im_original_scale = None

    def __getitem__(self, key):

        if key in ['current']:

            return self._current_fixture_settings

        elif key in ['fixture', 'reference']:

            return self._reference_fixture_settings

        else:

            raise KeyError(key)

    def set_image(self, image=None, image_path=None):

        if image is not None:

            self.im = image

        elif image_path is not None:

            self.im = imread(image_path)
            if self.im is None:
                self._logger.error("Could not load image")

            else:
                self._logger.info("Loaded image {0} with shape {1}".format(image_path, self.im.shape))
        else:

            self._logger.warning("No information supplied about how to load image, thus none loaded")

            self.im = None

        self.im_original_scale = get_image_scale(self.im)

    def analyse_current(self):

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

    def _get_markings(self, source='reference'):

        markers = self[source].get_marker_positions()

        if markers:

            return np.array(markers[0]), np.array(markers[1])

        return None, None

    def run_marker_analysis(self):

        _logger = self._logger

        t = time.time()

        _logger.debug("Scaling image")

        analysis_img = self._get_image_in_correct_scale()

        _logger.debug("Setting up Image Analysis (acc {0} s)".format(time.time() - t))

        im_analysis = imageFixture.FixtureImage(
            image=analysis_img,
            pattern_image_path=self["reference"].get_marker_path(),
            scale=self['reference'].model.scale)

        _logger.debug("Finding pattern (acc {0} s)".format(time.time() - t))

        x_positions, y_positions = im_analysis.find_pattern(self["reference"].get_marker_positions())

        self["current"].model.orientation_marks_x = x_positions
        self["current"].model.orientation_marks_y = y_positions

        if x_positions is None or y_positions is None:

            _logger.error("No markers found")

        _logger.debug("Marker Detection complete (acc {0} s)".format(time.time() - t))

    def _get_image_in_correct_scale(self):

        scale = self['reference'].model.scale
        if scale is None or scale == self.im_original_scale:
            analysis_img = self.im
        else:
            analysis_img = imageBasics.Quick_Scale_To_im(
                im=self.im,
                scale=self.MARKER_DETECTION_SCALE / self.im_original_scale)

        return analysis_img

    def _set_current_mark_order(self):

        x_centered, y_centered = self._get_centered_mark_positions("current")
        x_centered_ref, y_centered_ref = self._get_centered_mark_positions("reference")

        if x_centered and y_centered and x_centered_ref and y_centered_ref:

            length = np.sqrt(x_centered ** 2 + y_centered ** 2)
            length_ref = np.sqrt(x_centered_ref ** 2 + y_centered_ref ** 2)

            sort_order, sort_error = self._get_sort_order(length, length_ref)

            self._logger.debug(
                "Found sort order that matches the reference {0} (error {1})".format(sort_order, sort_error))

            self.__set_current_mark_order(sort_order)

        else:

            self._logger.critical("Missmatch in number of markings!")


    def __set_current_mark_order(self, sort_order):

        current_model = self["current"].model
        current_model.orientation_marks_x = current_model.orientation_marks_x[sort_order]
        current_model.orientation_marks_y = current_model.orientation_marks_y[sort_order]

    def _get_centered_mark_positions(self, source="current"):

        x_positions = self[source].model.orientation_marks_x
        y_positions = self[source].model.orientation_marks_y

        if x_positions is None or y_positions is None:
            return None, None

        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)

        marking_center = np.array((x_positions.mean(), y_positions.mean()))

        x_centered = x_positions - marking_center[0]
        y_centered = y_positions - marking_center[1]

        return x_centered, y_centered

    @staticmethod
    def _get_sort_order(length, reference_length):

        s = range(len(length))

        tmp_dL = []
        tmp_s = []
        for i in itertools.permutations(s):

            tmp_dL.append((length[list(i)] - reference_length) ** 2)
            tmp_s.append(i)

        dLs = np.array(tmp_dL).sum(1)
        return list(tmp_s[dLs.argmin()]), np.sqrt(dLs.min())

    @staticmethod
    def _get_rotations(dX, dY, L, ref_dX, ref_dY, ref_L, s):

        A = np.arccos(dX / L)
        A = A * (dY > 0) + -1 * A * (dY < 0)

        ref_A = np.arccos(ref_dX / ref_L)
        ref_A = ref_A * (ref_dY > 0) + -1 * ref_A * (ref_dY < 0)

        dA = A[s] - ref_A
        """:type : numpy.array"""
        return dA.mean()

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


    def _get_relative_point(self, point, alpha=None, offset=(0, 0)):
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

        self._set_current_mark_order()
        X, Y = self._get_markings(source='current')
        ref_Mcom = np.array(self['fixture']["marking_center_of_mass"])

        dMcom = Mcom - ref_Mcom

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
                [self._get_relative_point(Gs1, alpha, offset=dMcom),
                 self._get_relative_point(Gs2, alpha, offset=dMcom)])

        else:

            if bool(self['fixture']['grayscale']) is not True:

                self._logger.warning("No grayscale enabled in reference")

            if ref_gs is None:

                self._logger.warning("No grayscale area in reference")

        i = 0

        p_str = "plate_{0}_area"
        f_plates = self['fixture'].get_all("plate_%n_area")

        for i, p in enumerate(f_plates):
            p = np.array(p)
            p *= self.im_original_scale
            M1 = scale_factor * p[0]
            M2 = scale_factor * p[1]

            self['current'].set(p_str.format(i),
                [self._get_relative_point(M1, alpha, offset=dMcom),
                 self._get_relative_point(M2, alpha, offset=dMcom)])


    def _set_current_areas(self):

        self._set_current_mark_order()

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
                [self._get_relative_point(dGs1, alpha, offset=Mcom),
                 self._get_relative_point(dGs2, alpha, offset=Mcom)])

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
                [self._get_relative_point(dM1, alpha, offset=Mcom),
                 self._get_relative_point(dM2, alpha, offset=Mcom)])

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
                    p = _get_coords_sorted(p)
                    plate_list.append(p)

            i += 1

        return plate_list