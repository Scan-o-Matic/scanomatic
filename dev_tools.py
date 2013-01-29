#!/usr/bin/env python

import os
from matplotlib import pyplot as plt

import src.analysis_image as analysis_image
import src.resource_grid as resource_grid
import src.analysis_grid_array as analysis_grid_array

def get_plate(im=None, im_path=None, plate_index=0, log_version=1.0, run_insane=True):

    if im is None:
        if os.path.isfile(im_path):
            try:
                im = plt.imread(im_path)
            except:
                pass

    if im is None:
        return None


    if im_path is not None:
        basename = os.path.basename(im_path)
        dirname = os.path.dirname(im_path)
    else:
        basename = None
        dirname = '.'

    pm = [(32, 48)] * 4

    project_image = analysis_image.Project_Image(
                pm,
                verbose=True,
                file_path_base=dirname,
                fixture_name=None,
                logger=None,
                log_version=log_version
                )
    plates = project_image.fixture.get_plates('fixture')
    print "Plates" , plates, "in", project_image.fixture['name']
    if plate_index < len(plates):
        features = plates[plate_index]
    else:
        return None

    #features = project_image.fixture['fixture'][plate]
    print features

    return project_image.get_im_section(features, scale_factor=1, im=im, run_insane=run_insane)
    

def get_thresholded_plate(p):

    T = resource_grid.get_adaptive_threshold(p)
    tim = p < T

    ftim = resource_grid.get_denoise_segments(tim)
    resource_grid.get_segments_by_size(ftim, 40, 54*54)
    resource_grid.get_segments_by_shape(ftim, (54, 54))

    return ftim


def get_blob_centra(p):

    return resource_grid.get_blob_centra(p)


def show_grid(p, grid, X=None, Y=None):


    aga = analysis_grid_array.Grid_Array(None, [None, None],
        tuple(sorted(grid.shape[:-1])))

    aga.set_manual_grid(grid)
    aga.make_grid_im(p, X=X, Y=Y)
