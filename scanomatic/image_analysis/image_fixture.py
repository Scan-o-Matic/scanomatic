import os
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass

#
# INTERNAL DEPENDENCIES
#
from image_basics import load_image_to_numpy
import scanomatic.io.logger as logger
import image_basics

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

        self._logger.info("Analysing image {0} using pattern file {1} and scale {2}".format(
            path if path else (image.shape if image is not None else "NO IMAGE"), pattern_image_path, scale))

        if os.path.isfile(pattern_image_path) is False and resource_paths is not None:

            pattern_image_path = os.path.join(resource_paths.images, os.path.basename(
                pattern_image_path))

        if pattern_image_path:

            try:

                pattern_img = load_image_to_numpy(pattern_image_path, dtype=np.uint8)

            except IOError:

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

                    self._img = load_image_to_numpy(path, np.uint8)

                except IOError:

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

            self._img = load_image_to_numpy(path, dtype=np.uint8)

        except IOError:

            self._logger.error("Could not reload image at " + str(path))
            self._load_error = True

        if self._load_error is not True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]

        if conversion_factor != 1.0:

            self._logger.info("Scaling to {0}".format(conversion_factor))
            self._img = image_basics.Quick_Scale_To_im(conversion_factor)
            self._logger.info("Scaled")

    @staticmethod
    def get_hit_refined(local_hit, conv_img, coordinates=None, gaussian_weight_size_fraction=2.0):
        """
        Use half-size to select area and give each pixel the weight of the convolution result of
        that coordinate times the 2D gaussian value based on offset of distance to hit (sigma = ?).

        Refined hit is the hit + mass-center offset from hit

        :param hit:
        :param conv_img:
        :param half_stencil_size:
        :return: refined hit
        :rtype : (int, int)
        """

        def make_2d_guass_filter(size, fwhm, center):
            """ Make a square gaussian kernel.

            size is the length of a side of the square
            fwhm is full-width-half-maximum, which
            can be thought of as an effective radius.
            """

            x = np.arange(0, size, 1, float)
            y = x[:,np.newaxis]

            x0 = center[1]
            y0 = center[0]

            return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

        if coordinates is None:
            image_slice = conv_img
        else:
            image_slice = conv_img[int(round(coordinates['d0_min'])): int(round(coordinates['d0_max'])),
                                   int(round(coordinates['d1_min'])): int(round(coordinates['d1_max']))]

        gauss_size = max(image_slice.shape)

        gauss = make_2d_guass_filter(gauss_size, gauss_size / gaussian_weight_size_fraction, local_hit)

        return np.array(center_of_mass(image_slice * gauss[: image_slice.shape[0], : image_slice.shape[1]])) - \
            local_hit

    def get_convolution(self, threshold=127):

        t_img = (self._img > threshold).astype(np.int8) * 2 - 1
        marker = self._pattern_img

        if len(marker.shape) == 3:

            marker = marker[:, :, 0]

        t_mrk = (marker > 0) * 2 - 1

        return fftconvolve(t_img, t_mrk, mode='same')

    @staticmethod
    def get_best_location(conv_img, stencil_size, refine_hit_gauss_weight_size_fraction=2.0,
                          max_refinement_iterations=20, min_refinement_sq_distance=0.0001):
        """This whas hidden and should be taken care of, is it needed"""

        hit = np.where(conv_img == conv_img.max())

        if len(hit[0]) == 0:

            return None, conv_img

        hit = np.array((hit[0][0], hit[1][0]), dtype=float)

        #Zeroing out hit
        half_stencil_size = map(lambda x: x / 2.0, stencil_size)

        coordinates = {'d0_min': max(0, hit[0] - half_stencil_size[0] - 1),
                       'd0_max': min(conv_img.shape[0], hit[0] + half_stencil_size[0]),
                       'd1_min': max(0, hit[1] - half_stencil_size[1] - 1),
                       'd1_max': min(conv_img.shape[1], hit[1] + half_stencil_size[1])}

        for _ in range(max_refinement_iterations):

            offset = FixtureImage.get_hit_refined(
                    hit - (coordinates['d0_min'], coordinates['d1_min']), conv_img, coordinates,
                    refine_hit_gauss_weight_size_fraction)

            hit += offset

            if (offset ** 2).sum() < min_refinement_sq_distance:
                break

        coordinates = {'d0_min': int(round(max(0, hit[0] - half_stencil_size[0] - 1))),
                       'd0_max': int(round(min(conv_img.shape[0], hit[0] + half_stencil_size[0]))),
                       'd1_min': int(round(max(0, hit[1] - half_stencil_size[1] - 1))),
                       'd1_max': int(round(min(conv_img.shape[1], hit[1] + half_stencil_size[1])))}

        conv_img[coordinates['d0_min']: coordinates['d0_max'],
                 coordinates['d1_min']: coordinates['d1_max']] = conv_img.min() - 1

        return hit, conv_img


    def get_best_locations(self, conv_img, stencil_size, n, refine_hit_gauss_weight_size_fraction=2.0):
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

            m_loc, c_img = self.get_best_location(
                c_img, stencil_size, refine_hit_gauss_weight_size_fraction)

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
                markings,
                refine_hit_gauss_weight_size_fraction=3.5)) * self._conversion_factor

            try:
                return m1[:, 1], m1[:, 0]
            except (IndexError, TypeError):
                self._logger.error("Detecting markings failed, location object:\n{0}".format(m1))
                return None, None

        else:

            return None, None

    def get_loaded(self):

        return (self._img is not None) and (self._load_error is not True)
