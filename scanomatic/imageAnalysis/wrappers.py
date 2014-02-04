#!/usr/bin/env python
"""Wrapper for import of various aspects of the analysis procedure"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

#import numpy as np
import random
import os

#
# SCANNOMATIC LIBRARIES
#

import grid_array
import grid_cell
import analysis as project
import analysis_image
import scanomatic.io.project_log as project_log
import support
import scanomatic.io.paths as paths

#
# Globals
#

_pi = analysis_image.Project_Image([])
_ga = grid_array.Grid_Array(_pi, (0,), None)
POLY = _ga.get_calibration_polynomial_coeffs()
_paths = paths.Paths()


#
# Functions
#

def get_grid_cell_from_first_pass_file(
        fpath, image=None, plate=None, cell=None):
    """Returns a dictionary with information sufficient to be used as
    keyword argument for calling the method again and the requested grid cell
    """
    meta_data = project_log.get_meta_data(
        path=fpath)

    if image is not None:
        if not(0 <= image <= meta_data['Measures']):
            raise Exception('Image {0} out of range (0 - {1})'.format(
                image, meta_data['Measures']))

    else:

        image = random.randint(0, meta_data['Measures'])

    if plate is not None:
        if (0 <= plate < len(meta_data['Pinning Matrices']) or
                meta_data['Pinning Matrices'][plate] is None):

            raise Exception('Plate {0} does not exist'.format(plate))

    else:

        plate = random.choice([
            i for i, p in enumerate(meta_data['Pinning Matrices'])
            if p is not None])

    if cell is not None:
        if (False in [0 <= v < meta_data['Pinning Matrices'][plate][i] for
                      i, v in enumerate(cell)]):

            raise Exception('Cell {0} out of plate range ({1})'.format(
                cell, meta_data['Pinning Matrices'][plate]))

    else:

        cell = [random.randint(0, v - 1) for v in
                meta_data['Pinning Matrices'][plate]]

    file_path_base = os.path.dirname(fpath)

    project_image = analysis_image.Project_Image(
        meta_data['Pinning Matrices'],
        file_path_base=file_path_base,
        fixture_name=meta_data['Fixture'],
        p_uuid=meta_data['UUID'],
        suppress_analysis=True,
        log_version=meta_data['Version'],
        paths=_paths)

    image_dicts = project_log.get_image_entries(path=fpath)
    image_dict = image_dicts[image]
    graph_watch = [plate] + cell
    nPlates, active_position_keys = support.get_active_plates(
        meta_data, True, graph_watch)
    graph_watch[0] = 0

    project_image.get_analysis(
        image_dict['File'],
        (image_dict[active_position_keys[0]],),
        grayscaleSource=image_dict['grayscale_values'],
        grayscaleTarget=image_dict['grayscale_indices'],
        watch_colony=graph_watch,
        identifier_time=image,
        image_dict=image_dict)

    return ({'fpath': fpath, 'image': image, 'plate': plate, 'cell': cell},
            project_image[0][cell])


def get_grid_cell_from_array(
        arr, fallback_detection=False, center=None,
        radius=None, invoke_transform=False):
    """
        get_grid_cell_from_array is a convinience function to pass a section
        of an image as argument and make the entire array be the grid_cell.

        Function takes argument:

        @arr    An numpy array containing image data of interest
                The entire array will be treated as being the grid cell


        @fallback_detection     If true (not default) will only use otsu.

        @center                 A manually set blob centrum (if set
                                radius must be set as well)
                                (if not supplied, blob will be detected
                                automatically)

        @radius                 A manually set blob radus (if set
                                center must be set as well)
                                (if not supplied, blob will be detected
                                automatically)

        Function returns a Grid_Cell instance that is ready to use.

    """

    global POLY

    if invoke_transform:

        poly = POLY

    else:

        poly = None

    settings = {'data_source': arr, 'no_analysis': True,
                'no_detect': (center is None or radius is None),
                'blob_detect': 'default', 'remember_filter': False,
                'polynomial_coeffs': poly}

    cell = grid_cell.Grid_Cell((0, 0, 0),
                               grid_cell_settings=settings)

    cell.attach_analysis(center=center, radius=radius)

    return cell


def get_grid_cell_analysis_from_array(arr):
    """
        get_grid_cell_analysis_from_array is a convenience function that
        allows passing a section of an image as argument and retrieving
        the analysed results.

        Function takes argument:

        @arr    An numpy array containing image data of interest
                The entire array will be treated as being the grid cell


        Function returns a features diectionary.

    """

    cell = get_grid_cell_from_array(arr)

    return cell.get_analysis()


'''
def get_gray_scale_transformation_matrix(gs_values):
    """
        get_gray_scale_transformation_matrix takes a list of gs-values and
        returns a transformation matrix for a normal 8-bit gray scale image.

    """

    arr = grid_array.Grid_Array(None, (0,) ,None)

    return arr.get_transformation_matrix(gs_values=gs_values,
        gs_indices = np.asarray([82,78,74,70,66,62,58,54,50,46,42,38,34,30,26,
            22,18,14,10,6,4,2,0]))
'''

#
# WRAPPER CLASSES
#


class Grid_Array(grid_array.Grid_Array):
    def __init__(self, root):
        grid_array.Grid_Array.__init__(self, root, (0,))


class Grid_Cell(grid_cell.Grid_Cell):
    def __init__(self):
        grid_cell.Grid_Cell.__init__(self, (0, 0, 0))


class Project_Image(analysis_image.Project_Image):
    def __init__(self, im_path):
        project.Project_Image.__init__(self, im_path)


#
# COMMAND LINE BEHAVIOUR
#

if __name__ == "__main__":
    pass
