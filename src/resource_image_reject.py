#!/usr/bin/env python
"""
This module is used when aquiring images to verify that the exposure and
detection is similar to previous images.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.0995"
_maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

from PIL import Image
import numpy as np
import sys, os
import types

#
# GLOBALS
#

_histograms = {}

#
# FUNCTIONS
#

def evaluate_images(image_list=None, log_file=None, manual_value=None, threshold=None, threshold_less_than=True, max_value=255):

    excepted_images = 0
    rejected_images = 0
    try:
        max_value = float(max_value)
    except:
        max_value = 255.0

    if not threshold:
        threshold = 1.0

    pre_results = []
    if log_file:
        try:
            fs = open(log_file,'r')
        except:
            print "*** Error: could not open", log_file        

        image_list = []
                
        for line in fs:
            try:
                image_list.append(eval(line.strip()))
            except:
                print "*** Error: Could not interpret:",str(line.strip())
        fs.close()

    count = 0
    for im_path in image_list:
        if not log_file:
            im_open = False
            try:
                im = Image.open(im_path)
                im_open = True
            except:
                im_open = False
                excepted_images += 1
        else:
            if 'Histogram' in im_path.keys():
                histogram = im_path['Histogram']
                try:
                    pixels = im_path['ImageLength'] * im_path['ImageWidth']
                except:
                    pass
                im_open = True
            else:
                try:
                    pixels = im_path['ImageLength'] * im_path['ImageWidth']
                except:
                    pass
                im_open = False
  
        if im_open:
            if not log_file:
                histogram = im.histogram()
                pixels = im.size()[0] * im.size()[1]
            p_sum = 0
            p_counted = 0
            p_max = 0
            p_max_arg = 0
            p_median = -1
            for i, value in enumerate(histogram):
                p_sum += i * value
                if value > p_max:
                    p_max = value
                    p_max_arg = i

                p_counted += value
                if p_counted >= pixels/2.0 and p_median < 0:
                    p_median = i

            if log_file:
                pre_results.append({'File':str(im_path['File']), 'Max Peak':p_max_arg, 'Median Value': p_median, 'Source Index': count})
            else:
                pre_results.append({'File':str(im_path), 'Max Peak':p_max_arg, 'Median Value': p_median, 'Source Index': count})
            count += 1

    print "*** Evaluated " + str(len(image_list)-excepted_images) + " images, "+str(excepted_images)+" were excluded/not found"

    if manual_value:
        ref_value = manual_value

    else:

        median_list = []
        for k in xrange(len(pre_results)):
            median_list.append(pre_results[k]['Median Value'])

        ref_value = np.median(np.array(median_list)) 
        
    print "*** Target value set at:", ref_value

    filtered_results = [] 
    for k in xrange(len(pre_results)):
        quality = 1 - abs(pre_results[k]['Median Value'] - ref_value)/max_value

        if bool(quality <= threshold) == bool(threshold_less_than == True):
            pre_results[k]['Quality'] = quality
            filtered_results.append(pre_results[k])

    print "*** " + str(len(filtered_results)) + " were filtered out by threshold."
    return filtered_results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "-h":

            manual_value = None
            slice_start = 1
            log_file = None
            threshold = None

            if "-v" in sys.argv:
                for i,a in enumerate(sys.argv):
                    if a == "-v" and type(sys.argv[i+1]) == types.IntType:
                        manual_value = int(sys.argv[i+1])
                        slice_start = i+1

            if "-l" in sys.argv:
                for i,a in enumerate(sys.argv):
                    if a == "-l":
                        log_file = sys.argv[i+1]
                        slice_start = None
                        
            if "-t" in sys.argv:
                for i,a in enumerate(sys.argv):
                    if a == "-t":
                        try:
                            threshold = float(sys.argv[i+1])
                        except:
                            pass

            if slice_start:
                print evaluate_images(file_list=sys.argv[slice_start:], manual_value = manual_value, threshold=threshold)
            elif log_file:
                print evaluate_images(log_file=log_file, manual_value = manual_value, threshold=threshold)

    else:
            print "You need to run this script with the file you want to convert as argument"
            print "COMMAND:",sys.argv[0], "[OPTIONS] [IMAGE-FILES]"
            print "\n\nOPTIONS:"
            print "-v xxx\t\tSets manual value (0-255) to which all images are compared."
            print "-l [PATH]\t\tUses a log-file instead of a filelist"
            print "-t x.xx\t\tSets a quality threshold below which image is reported. Default 1.0"
            sys.exit(0)
