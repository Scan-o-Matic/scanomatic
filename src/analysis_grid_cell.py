#!/usr/bin/env python
"""
Part of the analysis work-flow that holds the grid-cell object (A tile in a
grid-array with a potential blob at the center).
"""
__author__ = "Martin Zackrisson, Mats Kvarnstroem"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
#import math
#import logging

#
# SCANNOMATIC LIBRARIES
#

import analysis_grid_cell_dissection as cell_dissection

#
# CLASS: Grid_Cell
#


class Grid_Cell():

    #Limits for number of cells in cell-space
    MAX_THRESHOLD = 2500
    MIN_THRESHOLD = 0

    def __init__(self, parent, identifier, grid_cell_settings=None):

        self._parent = parent
        self.logger = self._parent.logger

        self._identifier = identifier
        self._adjustment_warning = False

        default_settings = {
            'data_source': None, 'no_analysis': False, 'no_detect': False,
            'blob_detect': 'default', 'remember_filter': True,
            'polynomial_coeffs': None}

        if grid_cell_settings is None:

            grid_cell_settings = default_settings

        for k in default_settings.keys():

            if k in grid_cell_settings.keys():

                setattr(self, k, grid_cell_settings[k])

            else:

                setattr(self, k, default_settings[k])

        self._previous_image = None

        self._analysis_items = {}

        self._analysis_item_names = ('blob', 'background', 'cell')

        for item_name in self._analysis_item_names:

            self._analysis_items[item_name] = None

    def __str__(self):

        s = "< {0}".format(self._identifier)

        if self.data_source is None:

            s += " No image set"

        else:

            s += " Image size: {0}".format(self.data_source.shape)

        s += " Layers: {0} >".format(self._analysis_items.keys())

        return s

    def __repr__(self):

        return self.__str__()

    #
    # SET functions
    #

    def set_data_source(self, data_source):

        self.data_source = data_source

    def get_overshoot_warning(self):

        return self._adjustment_warning

    def set_new_data_source_space(
            self, space='cell estimate',
            bg_sub_source=None, polynomial_coeffs=None):

        if space == 'cell estimate':

            self.logger.debug(
                "ANALYSIS GRID CELL: Kodak values ran" +
                " ({0} - {1})".format(self.data_source.min(),
                                      self.data_source.max()))

            if bg_sub_source is not None:

                bg_sub = np.mean(self.data_source[np.where(bg_sub_source)])
                self.logger.debug(
                    "ANALYSIS GRID CELL: Using " +
                    "{0} as background estimation {1} - {2})".format(
                        bg_sub,
                        (self.data_source[np.where(bg_sub_source)]).min(),
                        (self.data_source[np.where(bg_sub_source)]).max()))

                self.logger.debug(
                    "ANALYSIS GRID CELL: Good bg_sub_source = {0}".format(
                        bg_sub_source.max() == 1))

                self.data_source = self.data_source - bg_sub

            #MIN DETECTION THRESHOLD
            self.logger.debug(
                "ANALYSIS GRID CELL: Transforming -> " +
                "Cell Estimate, fixing negative cells counts ({0})".format(
                    np.where(self.data_source < 0)[0].size))
            self.data_source[self.data_source < self.MIN_THRESHOLD] = self.MIN_THRESHOLD

            #MAX DETECTION THRESHOLD
            self.logger.debug(
                "ANALYSIS GRID CELL: Transforming -> " +
                "Cell Estimate, fixing max overflow cells counts ({0})".format(
                    (self.data_source > self.MAX_THRESHOLD).sum()))
            max_detect_filter = self.data_source > self.MAX_THRESHOLD
            self._adjustment_warning = max_detect_filter.any()
            self.data_source[max_detect_filter] = self.MAX_THRESHOLD

            if polynomial_coeffs is not None:

                self.data_source = \
                    np.polyval(polynomial_coeffs, self.data_source)

            else:

                self.logger.warning(
                    "ANALYSIS GRID CELL: Was not fed any polynomial")

            self.logger.debug(
                "ANALYSIS GRID CELL: Cell Estimate values run" +
                " ({0} - {1})".format(self.data_source.min(),
                                      self.data_source.max()))

        self.set_grid_array_pointers()

    def set_grid_array_pointers(self):
        for item_names in self._analysis_items.keys():
            self._analysis_items[item_names].grid_array = self.data_source

    #
    # GET functions
    #

    def get_item(self, item_name):

        if item_name in self._analysis_items.keys():

            return self._analysis_items[item_name]

        else:

            return None

    def get_analysis(self, no_detect=None, no_analysis=None,
                     remember_filter=None, use_fallback=None):
        """get_analysis iterates through all possible cell items
        and runs their detect and do_analysis if they are attached.

        The cell items' features dictionaries are put inside a
        dictionary with the items' names as keys.

        If cell item is not attached, a None is put in the
        dictionary to avoid key errors.

        Function takes one optional argument:

        @no_detect      If set to true, it will re-use the
                        previously used detection.

        @no_analysis    If set to true, there will be no
                        analysis done, just detection (if still
                        active).

        @remember_filter    Makes the cell-item object remember the
                            detection filter array. Default = False.
                            Note: Only relevant when no_detect = False.

        @use_fallback       Optionally sets detection to iterative
                            thresholding"""

        if remember_filter is None:

            remember_filter = self.remember_filter

        features_dict = {}

        #This step only detects the objects
        self.get_item('blob').detect(remember_filter=remember_filter)
        self.get_item('background').detect()

        #Transfer data to 'Cell Estimate Space'
        bg_filter = self.get_item('background').filter_array

        if bg_filter.sum() == 0:

            self.logger.warning('Grid Cell {0}'.format(self._identifier) +
                                ' has no background (skipping)')

            return None

        else:

            self.set_new_data_source_space(
                space='cell estimate', bg_sub_source=bg_filter,
                polynomial_coeffs=self.polynomial_coeffs)

            for item_name in self._analysis_item_names:

                if self._analysis_items[item_name]:

                    self._analysis_items[item_name].\
                        set_data_source(self.data_source)

                    self._analysis_items[item_name].do_analysis()

                    features_dict[item_name] = \
                        self._analysis_items[item_name].features

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

        if blob:

            self._analysis_items['blob'] = cell_dissection.Blob(
                self, [self._identifier, ['blob']], self.data_source,
                blob_detect=blob_detect, run_detect=run_detect,
                center=center, radius=radius)

        if background and self._analysis_items['blob']:

            self._analysis_items['background'] = \
                cell_dissection.Background(
                    self, [self._identifier, ['background']], self.data_source,
                    self._analysis_items['blob'], run_detect=run_detect)

        if cell:
            self._analysis_items['cell'] = cell_dissection.Cell(
                self, [self._identifier, ['cell']], self.data_source,
                run_detect=run_detect)
