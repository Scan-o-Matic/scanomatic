"""Class for detecting fixture position in image"""

__author__ = "Martin Zackrisson, Andreas Skyman"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.994"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import imageBasics

#
# CLASSES
#


class FixtureImage(object):

    def __init__(self, path=None, image=None, pattern_image_path=None,
                 scale=1.0, resource_paths=None):

        self._path = path
        self._img = None
        self._pattern_img = None
        self._load_error = None
        self._transformed = False
        self._conversion_factor = 1.0 / scale
        self._logger = logger.Logger("Resource Image Analysis")

        if os.path.isfile(pattern_image_path) is False and resource_paths is not None:

            pattern_image_path = os.path.join(resource_paths.images, os.path.basename(
                pattern_image_path))

        if pattern_image_path:

            try:

                pattern_img = plt.imread(pattern_image_path)

            except:

                self._logger.error(
                    "Could not open orientation guide image at " +
                    str(pattern_image_path))

                self._load_error = True

            if self._load_error is not True:

                if len(pattern_img.shape) > 2:

                    pattern_img = pattern_img[:, :, 0]

                self._pattern_img = pattern_img

        if image is not None:

            self._img = np.asarray(image)

        if path:

            if not(self._img is None):

                self._logger.warning(
                    "Won't load from path since actually submitted")

            else:

                try:

                    self._img = plt.imread(path)

                except:

                    self._logger.error("Could not open image at " + str(path))
                    self._load_error = True

        if self._load_error is not True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]

    def load_other_size(self, path=None, conversion_factor=1.0):

        self._conversion_factor = float(conversion_factor)

        if path is None:

            path = self._path

        self._logger.info("Loading image from {0}".format(path))

        try:

            self._img = plt.imread(path)

        except:

            self._logger.error("Could not reload image at " + str(path))
            self._load_error = True

        if self._load_error is not True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]

        if conversion_factor != 1.0:

            self._logger.info("Scaling to {0}".format(conversion_factor))
            self._img = imageBasics.Quick_Scale_To_im(conversion_factor)
            self._logger.info("Scaled")

    def get_hit_refined(self, hit, conv_img):

        #quarter_stencil = map(lambda x: x/8.0, stencil_size)
        #m_hit = conv_img[(max_coord[0] - quarter_stencil[0] or 0):\
            #(max_coord[0] + quarter_stencil[0] or conv_img.shape[0]),
            #(max_coord[1] - quarter_stencil[1] or 0):\
            #(max_coord[1] + quarter_stencil[1] or conv_img.shape[1])]

        #mg = np.mgrid[:m_hit.shape[0],:m_hit.shape[1]]
        #w_m_hit = m_hit*mg *m_hit.shape[0] * m_hit.shape[1] /\
            #float(m_hit.sum())
        #refinement = np.array((w_m_hit[0].mean(), w_m_hit[1].mean()))
        #m_hit_max = np.where(m_hit == m_hit.max())
        #print "HIT: {1}, REFINED HIT: {0}, VECTOR: {2}".format(
            #refinement,
            #(m_hit_max[0][0], m_hit_max[1][0]),
            #(refinement - np.array([m_hit_max[0][0], m_hit_max[1][0]])))

        #print "OLD POS", max_coord, " ",
        #max_coord = (refinement - np.array([m_hit_max[0][0],
            #m_hit_max[1][0]])) + max_coord
        #print "NEW POS", max_coord, "\n"

        return hit

    def get_convolution(self, threshold=127):

        t_img = (self._img > threshold).astype(np.int8) * 2 - 1
        marker = self._pattern_img

        if len(marker.shape) == 3:

            marker = marker[:, :, 0]

        t_mrk = (marker > 0) * 2 - 1

        return fftconvolve(t_img, t_mrk, mode='same')

    def get_best_location(self, conv_img, stencil_size, refine_hit=True):
        """This whas hidden and should be taken care of, is it needed"""

        max_coord = np.where(conv_img == conv_img.max())

        if len(max_coord[0]) == 0:

            return None, conv_img

        max_coord = np.array((max_coord[0][0], max_coord[1][0]))

        #Refining
        if refine_hit:
            max_coord = self.get_hit_refined(max_coord, conv_img)

        #Zeroing out hit
        half_stencil = map(lambda x: x / 2.0, stencil_size)

        d1_min = (max_coord[0] - half_stencil[0] > 0 and
                  max_coord[0] - half_stencil[0] or 0)

        d1_max = (max_coord[0] + half_stencil[0] < conv_img.shape[0]
                  and max_coord[0] + half_stencil[0] or conv_img.shape[0] - 1)

        d2_min = (max_coord[1] - half_stencil[1] > 0 and
                  max_coord[1] - half_stencil[1] or 0)

        d2_max = (max_coord[1] + half_stencil[1] < conv_img.shape[1]
                  and max_coord[1] + half_stencil[1] or conv_img.shape[1] - 1)

        conv_img[d1_min: d1_max, d2_min:d2_max] = conv_img.min() - 1

        return max_coord, conv_img

    def get_best_locations(self, conv_img, stencil_size, n, refine_hit=True):
        """This returns the best locations as a list of coordinates on the
        CURRENT IMAGE regardless of if it was scaled"""

        m_locations = []
        c_img = conv_img.copy()
        i = 0
        try:
            n = int(n)
        except:
            n = 3

        while i < n:

            m_loc, c_img = self.get_best_location(c_img, stencil_size,
                                                  refine_hit)

            m_locations.append(m_loc)

            i += 1

        return m_locations

    def find_pattern(self, markings=3, img_threshold=127):
        """This function returns the image positions as numpy arrays that
        are scaled to match the ORIGINAL IMAGE size"""

        if self.get_loaded():

            c1 = self.get_convolution(threshold=img_threshold)

            m1 = np.array(self.get_best_locations(
                c1, self._pattern_img.shape,
                markings, refine_hit=False)) * self._conversion_factor

            try:
                return m1[:, 1], m1[:, 0]
            except (IndexError, TypeError):
                self._logger.error("Detecting markings failed, location object:\n{0}".format(m1))
                return None, None

        else:

            return None, None

    def get_loaded(self):

        return (self._img is not None) and (self._load_error is not True)
