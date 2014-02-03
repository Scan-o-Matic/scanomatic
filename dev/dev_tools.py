#!/usr/bin/env python

import os
from matplotlib import pyplot as plt
import numpy as np

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


def show_new_grid(p, grid, X=None, Y=None):

    gX, gY = grid
    old_grid = np.c_[gX, gY].reshape(48, 32, 2, order='A')
    show_grid(p, old_grid, X=X, Y=Y)

def show_grid(p, grid, X=None, Y=None):


    aga = analysis_grid_array.Grid_Array(None, [None, None],
        tuple(sorted(grid.shape[:-1])))

    aga.set_manual_grid(grid)
    aga.make_grid_im(p, X=X, Y=Y)

def benchmark_algoritm(plate, remove_fraction=0.5, times=1000, box_size=(54, 54),
    grid_shape=(48, 32), run_dev=True):

    #Just to get X and Y
    grid, X, Y = resource_grid.get_grid(plate, box_size=box_size,
        grid_shape=grid_shape, run_dev=run_dev)
    
    results = list()
    fh = open('./dev/run2.test', 'w')
    for x in xrange(times):
        f_XY = np.random.random(X.shape) > remove_fraction 
        tX = X[f_XY]
        tY = Y[f_XY]

        center, spacings = resource_grid.get_grid_parameters_4(
                tX, tY, grid_shape, spacings=box_size, center=None)

        results.append(center + spacings)
        fh.write("{0}\n".format(results[-1]))

    fh.close()
    return results
