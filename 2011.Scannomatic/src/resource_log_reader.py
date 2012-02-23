#!/usr/bin/env python
"""This script produces analysis of inter-scan noise within a project."""

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

import matplotlib.pyplot as plt
import numpy as np
import sys, os
import types

#
# SCANNOMATIC DEPENDENCIES
#

import resource_image_reject as reject_script

#
# GLOBALS
#

_histograms = []

#
# FUNCTIONS
#

def count_histograms():
    return len(_histograms)

def load_data(path):

    global _histograms

    _histograms = []
    try:
        fs = open(path,'r')           
    except:
        print "Error, the file '" + str(sys.argv[1]) + "' does not exist."

    while True:
        line = fs.readline()
        if not line:
            break
        put_data_where_it_belongs(line)

    fs.close()

def put_data_where_it_belongs(data):

    global _histograms

    try:
        data = eval(str(data).strip())
    except:
        print "Non-interpretable line:", data

    if type(data) == types.DictType:
        if "Histogram" in data.keys():
            _histograms.append((str(data['File']), data['Histogram']))

def display_histograms(files="", draw_plot=True, mark_rejected=True, threshold=1.0, threshold_less_than=True, log_file=None, manual_value=None, max_value=255, save_path=None):

    global _histograms

    plot_data = None
    plot_indices = []

    if type(files) != types.ListType:
        files = [files]


    for f in files:
        if f == None:
            f = ""
        
        for k in xrange(len(_histograms)):
            if str(f) in _histograms[k][0]:
                if plot_data == None:
                    plot_data = np.array(_histograms[k][1])
                else:
                    plot_data = np.vstack((plot_data, np.array(_histograms[k][1])))
                plot_indices.append(k)

    #print plot_data.shape
    #print len(_histograms), plot_indices
    #print _histograms

    if mark_rejected:
            rejections = reject_script.evaluate_images(log_file=log_file, manual_value=manual_value, threshold=threshold, threshold_less_than=threshold_less_than, max_value=max_value)

    if plot_data != None and draw_plot:
        plt.plot(plot_data.T)
        plt.ylabel("count")
        plt.xlabel("inv pixel value")
        plt.legend(plot_keys)
    else:
        plt.imshow(plot_data)
        plt.ylabel("image number")
        plt.xlabel("inv pixel value")
        if mark_rejected and True == False: #HACK Cause it's ugly
            for r in rejections:
                if r['Source Index'] in plot_indices:
                    plt.plot(np.array(xrange(max_value+1)), r['Source Index']*np.ones(max_value+1, int), 'w-', alpha=190, linewidth=4)
        if save_path:
                plt.savefig(save_path)
                plt.clf()
    if mark_rejected:
            return rejections
    else:
            return plot_data

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "-h":

            load_data(sys.argv[-1])
            A = None
            do_histograms = True
            threshold = None
            manual_value  = None
            max_value = 255
            threshold_less_than = True

            if "-t" in sys.argv:
                for i in xrange(len(sys.argv)):
                    if sys.argv[i] == "-t":
                        try:
                            threshold = float(sys.argv[i+1])
                        except:
                            pass

            if "-v" in sys.argv:
                for i in xrange(len(sys.argv)):
                    if sys.argv[i] == "-v":
                        try:
                            manual_value = float(sys.argv[i+1])
                        except:
                            pass

            if "-m" in sys.argv:
                for i in xrange(len(sys.argv)):
                    if sys.argv[i] == "-m":
                        try:
                            max_value = int(sys.argv[i+1])
                        except:
                            pass

            if "-g" in sys.argv:
                threshold_less_than = False
                                    
            if "-i" in sys.argv:
                do_histograms = False

            if "-p" in sys.argv:
                start_slice = None
                stop_slice = None
                for i, a in enumerate(sys.argv[:-1]):
                    if not start_slice:
                        if a == "-p" and i+1 < len(sys.argv):
                            start_slice = i+1
                    elif not stop_slice:
                        if str(a)[0] == "-":
                            stop_slice = i+1

                if not stop_slice:
                    stop_slice = i+1
                if start_slice:
                    A = display_histograms(sys.argv[start_slice:stop_slice], do_histograms, log_file=sys.argv[-1], max_value=max_value, manual_value=manual_value, threshold=threshold, threshold_less_than=threshold_less_than)

            if "-i" in sys.argv:
                if A == None:
                    A = display_histograms("",do_histograms, log_file=sys.argv[-1], max_value=max_value, manual_value=manual_value, threshold=threshold, threshold_less_than=threshold_less_than)


            plt.show()

    else:
            print "You need to run this script with the file you want to convert as argument"
            print "COMMAND:",sys.argv[0],"[OPTIONS] [LOG-FILE]"
            print "\n\nOPTIONS:"
            print "-f [PATTERNS]\t\tPlots a histogram containing plots for all files in the log-file that matches a pattern"
            print "-i\t\t\tUses imshow to display a composite heatmap instead of histograms"
            print "-t x.xx\t\tSet exclusion threshold level (default = 1.0)"
            print "-g\t\t\tInvert threshold logic (default logic is to mark those less than threshold)"
            print "-m xxx\t\tSet pixel max value (default 255)"
            print "-v x.xx\t\tSet default target value for exclusion script manually (default is using median of medians)"
            sys.exit(0)
