#!/usr/bin/env python
"""
Part of the analysis work-flow that holds the grid-cell object (A tile in a
grid-array with a potential blob at the center).
"""
__author__ = "Martin Zackrisson, Mats Kvarnstroem"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
from scipy.stats.mstats import tmean, mquantiles

#
# SCANNOMATIC LIBRARIES
#

import grid_cell_extra as grid_cell_extra
from scanomatic.models.analysis_model import VALUES, ITEMS

#
# CLASS: Grid_Cell
#


class GridCell():

    MAX_THRESHOLD = 4200
    MIN_THRESHOLD = 0

    def __init__(self, identifier, analysis_model):

        self._identifier = identifier
        self.position = tuple(identifier[-1])
        self._analysis_model = analysis_model
        self._adjustment_warning = False
        self.xy1 = []
        self.xy2 = []
        self.source = None
        self.ready = False
        self._previous_image = None

        self._analysis_items = {}
        self._set_empty_analysis_items()

    def _set_empty_analysis_items(self):

        for item_name in ('blob', 'background', 'cell'):

            self._analysis_items[item_name] = None

    def __str__(self):

        s = "< {0}".format(self._identifier)

        if self.source is None:

            s += " No image set"

        else:

            s += " Image size: {0}".format(self.source.shape)

        s += " Layers: {0} >".format(self._analysis_items.keys())

        return s

    def __repr__(self):

        return self.__str__()

    def set_grid_coordinates(self, grid_cell_corners):

        self.xy1 = grid_cell_corners[:, self.position[0], self.position[1]]
        self.xy2 = grid_cell_corners[:, self.position[0] + 1, self.position[1] + 1]

    def get_overshoot_warning(self):

        return self._adjustment_warning

    def set_new_data_source_space(self, space=VALUES.Cell_Estimates, bg_sub_source=None, polynomial_coeffs=None):

        if space is VALUES.Cell_Estimates:

            if bg_sub_source is not None:

                feature_array = self.source[np.where(bg_sub_source)]
                bg_sub = tmean(feature_array,
                               mquantiles(feature_array, prob=[0.25, 0.75]))
                if not np.isfinite(bg_sub):
                    bg_sub = np.mean(feature_array)

                self.source -= bg_sub

            self.source[self.source < self.MIN_THRESHOLD] = self.MIN_THRESHOLD

            if polynomial_coeffs is not None:

                self.source = np.polyval(polynomial_coeffs, self.source)

            self._set_max_value_filter()

        self.push_source_data_to_cell_items()

    def _set_max_value_filter(self):

        max_detect_filter = self.source > self.MAX_THRESHOLD
        self._adjustment_warning = max_detect_filter.any()
        self.source[max_detect_filter] = self.MAX_THRESHOLD

    def push_source_data_to_cell_items(self):
        for item_names in self._analysis_items.keys():
            self._analysis_items[item_names].grid_array = self.source

    def get_item(self, item_name):

        if item_name in self._analysis_items.keys():

            return self._analysis_items[item_name]

        else:

            return None

    def get_analysis(self, no_detect=False, remember_filter=False):
        """get_analysis iterates through all possible cell items
        and runs their detect and do_analysis if they are attached.

        The cell items' features dictionaries are put inside a
        dictionary with the items' names as keys.

        If cell item is not attached, a None is put in the
        dictionary to avoid key errors.

        Function takes one optional argument:

        @no_detect      If true, it will re-use the
                        previously used detection.

        @remember_filter    Makes the cell-item object remember the
                            detection filter array. Default = False.
                            Note: Only relevant when no_detect = False.

        @use_fallback       Optionally sets detection to iterative
                            thresholding"""

        features_dict = {}

        if not no_detect:

            self.get_item('blob').detect(remember_filter=remember_filter)
            self.get_item('background').detect()

        bg_filter = self.get_item('background').filter_array

        if bg_filter.sum() == 0:

            features_dict = None

        else:

            self.set_new_data_source_space(space=VALUES.Cell_Estimates, bg_sub_source=bg_filter,
                polynomial_coeffs=self.polynomial_coeffs)

            for item_name in self._analysis_item_names:

                if self._analysis_items[item_name]:

                    self._analysis_items[item_name].set_data_source(self.source)
                    self._analysis_items[item_name].do_analysis()

                    features_dict[item_name] = self._analysis_items[item_name].features

                else:

                    features_dict[item_name] = None

        return features_dict

    #
    # Other functions
    #

    def attach_analysis(self, blob=True, background=True, cell=True,
                        blob_detect='default', run_detect=None, center=None,
                        radius=None):

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

        if blob_detect is None:

            blob_detect = self.blob_detect

        if run_detect is None:

            run_detect = not(self.no_detect)

        if cell:
            self._analysis_items['cell'] = grid_cell_extra.Cell(
                [self._identifier, ['cell']], self.source,
                run_detect=run_detect)

        if blob:

            self._analysis_items['blob'] = grid_cell_extra.Blob(
                [self._identifier, ['blob']], self.source,
                blob_detect=blob_detect, run_detect=run_detect,
                center=center, radius=radius)

        if background and self._analysis_items['blob']:

            self._analysis_items['background'] = \
                grid_cell_extra.Background(
                    [self._identifier, ['background']], self.source,
                    self._analysis_items['blob'], run_detect=run_detect)

        self.set_ready_state()

    def set_ready_state(self):

        self.ready = any(self._analysis_items.values())