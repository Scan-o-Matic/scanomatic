"""Resource module for handling basic images operations."""

import numpy as np
import scanomatic.io.logger as logger
from scipy.ndimage import zoom

try:
    import Image
except ImportError:
    from PIL import Image

from scanomatic.models.analysis_model import IMAGE_ROTATIONS

#
# GLOBALS
#

_logger = logger.Logger("Basic Image Utils")

#
# FUNCTIONS
#


def scale_16bit_to_8bit_range(data):

    return data / (2 ** 16 - 1.) * 255


def round_to_8bit(data):

    return np.round(data).astype(np.uint8)


def load_image_to_numpy(path, orientation=IMAGE_ROTATIONS.Portrait, dtype=np.float64):

    im = Image.open(path)
    data = np.array(im)

    if data.dtype == np.uint16:
        data = scale_16bit_to_8bit_range(data)

    data_orientation = IMAGE_ROTATIONS.Portrait if max(data.shape) == data.shape[0] else IMAGE_ROTATIONS.Landscape

    if data_orientation == orientation:
        if dtype is None or data.dtype == dtype:
            return data
        elif dtype == np.uint8:
            return np.round(data).astype(dtype)
        else:
            return data.astype(dtype)
    else:
        if dtype is None or data.dtype == dtype:
            return data.T
        elif dtype == np.uint8:
            return np.round(data).T.astype(dtype)
        else:
            return data.T.astype(dtype)


def Quick_Scale_To_im(path=None, im=None, source_dpi=600, target_dpi=150,
                      scale=None):

    if im is None:

        try:

            im = load_image_to_numpy(path, dtype=np.uint8)

        except:

            _logger.error("Could not open source")

            return -1

    if scale is None:
        scale = target_dpi / float(source_dpi)

    small_im = zoom(im, scale, order=1)

    return small_im


class Image_Transpose(object):

    def __init__(self, sourceValues=None, targetValues=None, polyCoeffs=None):

        self._logger = logger.Logger("Image Transpose")
        self._source = sourceValues
        self._target = targetValues
        self._polyCoeffs = polyCoeffs

        if (self._polyCoeffs is None and self._target is not None and
                self._source is not None):

            try:
                self._polyCoeffs = np.polyfit(self._source, self._target, 3)
            except Exception, e:
                self._logger.critical(
                    "Could not produce polynomial from source " +
                    "{0} and target {1}".format(self._source, self._target))

                raise e

        if self._polyCoeffs is not None:
            self._poly = np.poly1d(self._polyCoeffs)
        else:
            errorCause = ""
            if self._source is None:
                errorCause += "No source "
            if self._target is None:
                errorCause += "No target "
            if self._polyCoeffs is None:
                errorCause += "No Coefficients"
            raise Exception(
                "Polynomial not initiated; can't transpose image: {0}".format(
                    errorCause))

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def coefficients(self):
        return self._polyCoeffs

    @property
    def polynomial(self):
        return self._poly

    def __call__(self, im):

        return self._poly(im)
