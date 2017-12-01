"""
Part of the analysis work-flow that holds the grid-cell object (A tile in a
grid-array with a potential blob at the center).
"""

#
# DEPENDENCIES
#

import os

import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import scanomatic.image_analysis.grid_cell_extra as grid_cell_extra
from scanomatic.models.analysis_model import VALUES, COMPARTMENTS
from scanomatic.models.factories.analysis_factories import \
    AnalysisFeaturesFactory
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.generics.maths import mid50_mean as iqr_mean
#
# CLASS: Grid_Cell
#


class GridCell(object):

    MAX_THRESHOLD = 4200
    MIN_THRESHOLD = 0
    _logger = Logger("Grid Cell")

    def __init__(self, identifier, polynomial_coeffs, save_extra_data=False):

        self._identifier = identifier
        self.position = tuple(identifier[-1])
        self.save_extra_data = save_extra_data
        self._polynomial_coeffs = polynomial_coeffs
        self._adjustment_warning = False
        self.xy1 = []
        self.xy2 = []
        self.source = None
        self.ready = False
        self._previous_image = None
        self.image_index = -1
        self.features = AnalysisFeaturesFactory.create(
            index=tuple(self.position), data={})
        self._analysis_items = {}
        """:type: dict[scanomatic.models.analysis_model.COMPARTMENTS |scanomatic.image_analysis.grid_cell_extra.CellItem]"""
        self._set_empty_analysis_items()

    def _set_empty_analysis_items(self):

        for item_name in COMPARTMENTS:

            self._analysis_items[item_name] = None

    def __str__(self):

        text = "< {0}".format(self._identifier)

        if self.source is None:

            text += " No image set"

        else:

            text += " Image size: {0}".format(self.source.shape)

        text += " Layers: {0} >".format(self._analysis_items.keys())

        return text

    def __repr__(self):

        return self.__str__()

    def set_grid_coordinates(self, grid_cell_corners):
        """Set grid coordinates, flipping vertical (rows) axis so that the
        coordinate system is defined as right-handed x-y, not the native image
        left-handed y-x, with y reaching down."""

        flipped_long_axis_position = (
            grid_cell_corners.shape[2] - self.position[0] - 1)
        self.xy1 = grid_cell_corners[
            :, 0, flipped_long_axis_position, self.position[1]].astype(np.int)
        self.xy2 = grid_cell_corners[
            :, 1, flipped_long_axis_position, self.position[1]].astype(np.int)

    def set_new_data_source_space(self, space=VALUES.Cell_Estimates,
                                  bg_sub_source=None, polynomial_coeffs=None):

        if space is VALUES.Cell_Estimates:

            if bg_sub_source is not None:

                feature_array = self.source[np.where(bg_sub_source)]
                # bg_sub = tmean(feature_array,
                #                mquantiles(feature_array, prob=[0.25, 0.75]))
                bg_sub = iqr_mean(feature_array)
                if not np.isfinite(bg_sub):
                    bg_sub = np.mean(feature_array)
                    GridCell._logger.warning(
                        "{0} caused background mean ({1}) due to inf".format(
                            self._identifier, bg_sub))

                self.source = bg_sub - self.source

            self.source[self.source < self.MIN_THRESHOLD] = self.MIN_THRESHOLD

            if polynomial_coeffs is not None:

                self.source = np.polyval(polynomial_coeffs, self.source)

            self._set_max_value_filter()

        self.push_source_data_to_cell_items()

    def _set_max_value_filter(self):

        max_detect_filter = self.source > self.MAX_THRESHOLD

        if self._adjustment_warning != max_detect_filter.any():
            self._adjustment_warning = not self._adjustment_warning
            if self._adjustment_warning:
                self._logger.warning(
                    "{0} got {1} pixel-values overshooting {2}.".format(
                        self._identifier, max_detect_filter.sum(),
                        self.MAX_THRESHOLD) +
                    " Further warnings for this colony suppressed.")
            else:
                self._logger.info(
                    "{0} no longer have pixels that reach {1} depth.".format(
                        self._identifier, self.MAX_THRESHOLD))

    def push_source_data_to_cell_items(self):
        for item_names in self._analysis_items:
            self._analysis_items[item_names].grid_array = self.source

    def get_item(self, item_name):

        if item_name in self._analysis_items:

            return self._analysis_items[item_name]

        else:

            return None

    def analyse(self, detect=True, remember_filter=True):
        """get_analysis iterates through all possible cell items
        and runs their detect and do_analysis if they are attached.

        The cell items' features dictionaries are put inside a
        dictionary with the items' names as keys.

        If cell item is not attached, a None is put in the
        dictionary to avoid key errors..
        """

        background = self._analysis_items[COMPARTMENTS.Background]

        if detect:
            self.detect(remember_filter=remember_filter)

        if background.filter_array.sum() == 0:
            self.clear_features()
        else:
            self._analyse()

    def get_save_data_path(self, base_path):

        if base_path is None:
            base_path = Paths().log

        return os.path.join(base_path, "grid_cell_{0}_{1}_{2}".format(
            self.image_index, self._identifier[0][1], "_".join(
                map(str, self._identifier[-1][::-1]))
        ))

    def save_data_image(self, suffix="", base_path=None):

        base_path = self.get_save_data_path(base_path)
        np.save(base_path + suffix + ".image.npy", self.source)

    def save_data_detections(self, base_path=None):

        base_path = self.get_save_data_path(base_path)

        blob = self._analysis_items[COMPARTMENTS.Blob]
        background = self._analysis_items[COMPARTMENTS.Background]

        np.save(base_path + ".background.filter.npy", background.filter_array)
        np.save(base_path + ".image.cells.npy", background.grid_array)
        np.save(base_path + ".blob.filter.npy", blob.filter_array)
        np.save(base_path + ".blob.trash.current.npy", blob.trash_array)
        np.save(base_path + ".blob.trash.old.npy", blob.old_trash)

    def clear_features(self):

        for item in self._analysis_items.itervalues():

            if item:

                item.features.data.clear()

    def _analyse(self):

        background = self._analysis_items[COMPARTMENTS.Background]

        self.set_new_data_source_space(
            space=VALUES.Cell_Estimates,
            bg_sub_source=background.filter_array,
            polynomial_coeffs=self._polynomial_coeffs
        )

        for item in self._analysis_items.itervalues():

            if item:

                item.set_data_source(self.source)
                item.do_analysis()

    def detect(self, remember_filter=True):

        blob = self._analysis_items[COMPARTMENTS.Blob]
        background = self._analysis_items[COMPARTMENTS.Background]

        blob.detect(remember_filter=remember_filter)
        background.detect()

    def attach_analysis(self, blob=True, background=True, cell=True,
                        blob_detect=grid_cell_extra.BlobDetectionTypes.DEFAULT,
                        run_detect=False, center=None, radius=None):

        """attach_analysis connects the analysis modules to the Grid_Cell.

        Function has three optional boolean arguments:

        @blob           Attaches blob item (default)

        @background     Attaches background item (default)
                        Only possible if blob is attached

        @cell           Attaches cell item (default)

        @use_fallback_detection         Causes simple thresholding instead
                        of more sophisticated detection (default False)

        @run_detect     Causes the initiation to run detection

        @center         A manually set blob centrum (if set
                        radius must be set as well)
                        (if not supplied, blob will be detected
                        automatically)

       @radius          A manually set blob radus (if set
                        center must be set as well)
                        (if not supplied, blob will be detected
                        automatically)"""

        if cell:
            item = grid_cell_extra.Cell(
                (self._identifier, COMPARTMENTS.Total), self.source,
                run_detect=run_detect)
            self.features.data[item.features.index] = item.features
            self._analysis_items[item.features.index] = item

        if blob:

            item = grid_cell_extra.Blob(
                (self._identifier, COMPARTMENTS.Blob), self.source,
                blob_detect=blob_detect, run_detect=run_detect,
                center=center, radius=radius)

            self.features.data[item.features.index] = item.features
            self._analysis_items[item.features.index] = item

        if background and self._analysis_items[COMPARTMENTS.Blob]:

            item = grid_cell_extra.Background(
                (self._identifier, COMPARTMENTS.Background), self.source,
                self._analysis_items[COMPARTMENTS.Blob], run_detect=run_detect)

            self.features.data[item.features.index] = item.features
            self._analysis_items[item.features.index] = item

        self.features.shape = (len(self.features.data), )
        self.set_ready_state()

    def set_ready_state(self):

        self.ready = any(self._analysis_items.values())
