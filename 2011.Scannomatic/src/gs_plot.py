#!/usr/bin/env python
"""A script that produces a gray-scales plot from analysis.log file"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import os, sys
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import types
import numpy as np
from argparse import ArgumentParser


def make_plot_from_gs_list(gs_list, save_target=None):

    for Y in gs_list:

        X = range(len(Y))
        plt.plot(X,Y)

    plt.title("Showing grayscale calibrations for " + str(len(gs_list)) + \
        " images")
    if save_target:
        plt.savefig(save_target)
    else:
        plt.show()

    return True

def make_plot_from_logfile(file_path):

    try:
        fs = open(file_path, 'r')
    except:
        return False

    gs_list = []

    for line in fs:

        try:
            gs_list.append(eval(line.strip("\n"))['grayscale_values'])
        except:
            print "Warning: Ommiting line because it seems not to contain a GS"
            print line.strip("\n")

    fs.close()

    if gs_list == []:
        
        return False

    return make_plot_from_gs_list(gs_list)

if __name__ == "__main__":

    parser = ArgumentParser(description='The script plots all the grayscale ' + \
        'calibration values as a single plot...')

    parser.add_argument("-i", "--input-file", type=str, dest="inputfile", 
        help="Log-file to be parsed", metavar="PATH")

    args = parser.parse_args()

    if args.inputfile is not None:

        if not make_plot_from_logfile(args.inputfile):
            parser.error("Could not find specified file or contained no"+\
                " useful data")
        
    else:

        parser.error("Can't do much without an input file")
