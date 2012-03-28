#!/usr/bin/env python
"""
The script takes a list of scanned images and as far as possible creates a
logfile that can be used to run the analys work-flow.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.992"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

from PIL import Image
import sys, os
import types

#
# GLOBALS
#

_histograms = {}

#
# FUNCTIONS
#

def print_progress_bar(fraction, size=40):
    prog_str = "["
    fraction *= size
    for i in xrange(size):
        if fraction > i:
            prog_str += "="
        else:
            prog_str += " "

    prog_str += "]"

    print 
    print
    print prog_str

def make_entries(fs, file_list=None, extra_info=None, verboise=False, quiet=False):
    if file_list == None:
        if sys.argv[1][0] == "-":

            if sys.argv[1] == "-v":
                verboise = True 
            elif sys.argv[1] == "-q":
                quiet = True

            file_list = sys.argv[2:-1]
        else:
            file_list = sys.argv[1:-1]

        script_path_root = os.path.dirname(os.path.abspath(__file__))
        f_settings = settings_tools.Fixture_Settings(script_path_root + os.sep + "config" + os.sep + "fixtures", fixture="fixture_a")


    if type(file_list) != types.ListType:
        file_list = [file_list]

    exception_count = 0
    for f_i, im_file in enumerate(file_list):

        if verboise and not quiet:
            print_progress_bar(f_i/float(len(file_list)))
            print "** Processing:", im_file

        loaded_image = False
        try:
            im = Image.open(im_file)
            loaded_image = True
        except:
            print "Failed to analyse",im_file
            loaded_image = False
            exception_count += 1

        if loaded_image:
            
            #write_dictionary = {'File':str(im_file), 'Histogram':im.histogram(),'ImageLength':im.size[1] ,'ImageWidth':im.size[0]}
            write_dictionary = {'File':str(im_file)}
            if extra_info != None:
                for k,v in extra_info[f_i].items():
                    write_dictionary[k] = v
            else:
                #This means it was run from prompt and analysis needs to be done.
                f_settings.image_path = im_file
                f_settings.marker_analysis()
                f_settings.set_areas_positions()
                dpi_factor = 4.0
                f_settings.A.load_other_size(im_file, dpi_factor)
                grayscale = f_settings.A.get_subsection(f_settings.current_analysis_image_config.get("grayscale_area"))

                write_dictionary['mark_X'] = list(f_settings.mark_X)
                write_dictionary['mark_Y'] = list(f_settings.mark_Y)

                if grayscale != None:
                    gs = img_base.Analyse_Grayscale(image=grayscale, )
                    write_dictionary['grayscale_values'] = gs._grayscale
                    write_dictionary['grayscale_inices'] = gs._grayscale_X
                else:
                    write_dictionary['grayscale_values'] = None
                    write_dictionary['grayscale_inices'] = None
            
                sections_areas = f_settings.current_analysis_image_config.get_all("plate_%n_area")
                for i, a in enumerate(sections_areas):
                    #s = f_settings.A.get_subsection(a)
                    write_dictionary['plate_' + str(i) + '_area'] = list(a)

            if 'Time' not in write_dictionary.keys():
                write_dictionary['Time'] = os.stat(im_file).st_mtime

            fs.write(str(write_dictionary)+"\n\r")

    if verboise and not quiet:
        print_progress_bar(1)

    if not quiet:
        print "*** Log file made for "+str(len(file_list)-exception_count) + " images with "+str(exception_count)+" exceptions."

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] != "-h":
            try:
                fs = open(sys.argv[-1],'a')           
            except:
                fs = open(sys,argv[-1],'w')
                #print "Error, the file '" + str(sys.argv[-1]) + "' can not be created."

            import image_analysis_base as img_base
            import settings_tools as settings_tools

            make_entries(fs)

            fs.close()

    else:
            print "You need to run this script with the file you want to convert as argument"
            print "COMMAND:",sys.argv[0],"[-v/-q] [IMAGE-FILES] [LOG-FILE]\n\n"
            print "-v\t\tVerboise"
            print "-q\t\tQuiet"
            sys.exit(0)
