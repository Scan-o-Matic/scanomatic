import numpy as np
import itertools

#
#   INTERNAL DEPENDENCIES
#

import mock_numpy_interface


class SubPlates(mock_numpy_interface.NumpyArrayInterface):

    def __init__(self, dataObject, kernels=None):
        """This class puts an interchangeable subsampling level
        onto any applicable dataObject.

        If no kernel is set the layer is transparent and the original
        data in its original conformation can be directly accessed.

        If a kernel is in place, that plate will become strided such
        that it will expose an array as it would have looked before
        it got interleaved into the current plate.

        The

        Parameters:
            dataObject      An object holding several plates

            kernels         An array of kernels or None(s)

        """
        self._smooth_growth_data = dataObject
        self._kernels = None
        self.kernels = kernels

    @property
    def kernels(self):
        return self._kernels

    @kernels.setter
    def kernels(self, kernels):

        if (kernels is not None):

            assert len(kernels) == len(self._smooth_growth_data), (
                "Must have exactly as many kernels {0} as plates {1}".format(
                    len(kernels), len(self._smooth_growth_data)))

            for i, kernel in enumerate(kernels):

                if (kernel is not None):

                    assert kernel.sum() == 1, (
                        "All kernels must have exactly one true value "
                        "(kernel {0} has {1})".format(i, kernel.sum()))

                    assert np.array(
                        [p % k == 0 for p, k in itertools.izip(
                            self._smooth_growth_data[0].shape[:2],
                            kernel.shape)]).all(), (
                                "Dimension missmatch between kernel and plate"
                                " ({0} not evenly divisable with {1})".format(
                                    self._smooth_growth_data.shape[:2], kernel.shape))

        self._kernels = kernels

    def __getitem__(self, value):

        plate = self._smooth_growth_data[value]

        if (self._kernels is None or self._kernels[value] is None):
            return plate

        else:

            kernel = self._kernels[value]
            kernelD1, kernelD2 = (v[0] for v in np.where(kernel))

            assert plate.ndim in (2, 3), (
                "Plate {0} has wrong number of dimensions {1}".format(
                    value, plate.ndim))

            if (plate.ndim == 2):

                ravelOffset = plate.shape[1] * kernelD1 + kernelD2
                plateShape = (plate.shape[0] / kernel.shape[0],
                              plate.shape[1] / kernel.shape[1])
                plateStrides = (plate.strides[0] * kernel.shape[0],
                                plate.strides[1] * kernel.shape[1])

            elif (plate.ndim == 3):

                ravelOffset = (plate.shape[2] * plate.shape[1] * kernelD1 +
                               plate.shape[2] * kernelD2)
                plateShape = (plate.shape[0] / kernel.shape[0],
                              plate.shape[1] / kernel.shape[1],
                              plate.shape[2])
                plateStrides = (plate.strides[0] * kernel.shape[0],
                                plate.strides[1] * kernel.shape[1],
                                plate.strides[2])

            return np.lib.stride_tricks.as_strided(
                plate.ravel()[ravelOffset:],
                shape=plateShape,
                strides=plateStrides)
