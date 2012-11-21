#!/usr/bin/env python
"""Resource module for first pass analysis."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os

#
# INTERNAL DEPENDENCIES
#

import src.resource_logger as resource_logger

def analyse(file_name, fixture, logger=None):

    analysis = dict()
    if logger is None:
        logger = resource.Log_Garbage_Collector()

    """
            im_data = {'Time':time.time(), 'File': f}
            self.DMS("Analysis", "Grayscale analysis of" + str(f), 
                level="LA", debug_level='debug')

            self.f_settings.set_image(image_path=f)

            self.f_settings.run_marker_analysis()
            self.f_settings.set_current_areas()
            self.f_settings.analyse_grayscale()

            gs_indices, gs_values = self.f_settings['grayscale']
            
            im_data['grayscale_values'] = gs_values
            im_data['grayscale_indices'] = gs_indices
            im_data['mark_X'], im_data['mark_Y'] = self.f_settings['markers']            

            sections_areas = self.f_settings['plates']

            im_data['plates'] = len(sections_areas)

            plate_str = "plate_{0}_area"
            for i, a in enumerate(sections_areas):
                im_data[plate_str.format(i)] = list(a)

            data.append(im_data)
    """

    return dict()

