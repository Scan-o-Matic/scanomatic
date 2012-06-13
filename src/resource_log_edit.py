#!/usr/bin/env python
"""
The module rewrites logfiles created as projects are running.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.993"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import logging, uuid, os

#
# GLOBALS
#

META_DATA = {'Start Time': 0, 'Prefix': 'unknown', 'Interval': 20.0, 
   'Description': 'Automatic placeholder description',
   'UUID': str(uuid.uuid1()),'Measures':0,'Fixture': 'fixture_a',
   'Pinning Matrices': None, 'Manual Gridding': None}

#
# FUNCTIONS
#

def rewrite_meta_row(path, header_entries):

    try:
        fs = open(path, 'r')
    except:
        logging.error("Could not open '{0}'".format(path))
        return False


    lines = []
    i = 0
    for l in fs:
        if i == 0:
            l_dict = eval(l)
            try:
                l_keys_in_meta = [(k in META_DATA.keys()) for k in l_dict.keys()]
            except:
                l_keys_in_meta = [False]

            if len(l_keys_in_meta) > sum(l_keys_in_meta):
                data = l_dict
            else:
                data = META_DATA

            for h in header_entries.keys():
                data[h] = header_entries[h]
             
            if len(l_keys_in_meta) > sum(l_keys_in_meta):
                lines = [str(data)+"\n", l]
            else:
                lines = [str(data)+"\n"]
        else:
            lines.append(l)
        i += 1
    fs.close()

    fs = open(path, 'w')
    for l in lines:
        fs.write(l)
    fs.close()
    return True

def rewrite(path, rewrite_entries):

    try:
        fs = open(path, 'r')
    except:
        logging.error("Could not open '{0}'".format(path))
        return False


    lines = []

    i = 0
    for l in fs:
        if i in rewrite_entries.keys():
            lines.append(str(rewrite_entries[i]['data']))
            if rewrite_entries[i]['mode'] == 'a':
                lines.append(l)
        else:
            lines.append(l)
        i += 1
    fs.close()

    fs = open(path, 'w')

    for l in lines:
        fs.write(l)

    fs.close()
    return True

def create_place_holder_meta_info(path = None):

    data = META_DATA

    if path:

        try:

            fs = open(path, 'r')
            file_exists = True

        except: 

            logging.warning("The file at '{0}' could not been opened".format(path))
            file_exists = False

        if file_exists:

            lowest_time = None
            n_images = 0
            prefix = None

            for line in fs:
                try:
                    l = eval(line)
                except:
                    l = {}

                try:
                    'mark_X' in l.keys()
                    good_line = True
                except:
                    loggin.warning("The file '{0}' contains unexpected line '{1}'"\
                        .format(path, line))
                    good_line = False

                if good_line: 
                    if 'Time' in l.keys():
                        if lowest_time is None or l['Time'] < lowest_time:
                            lowest_time = l['Time']

                    if prefix is None and 'File' in l.keys():
                        try:
                            prefix = l['File'].split(os.sep)[-1][:-10]
                        except:
                            logging.warning('Problem guessing the original prefix')

                    if 'mark_Y' in l.keys():

                        n_images += 1
            fs.close()

            if prefix is not None:
                data['Prefix'] = prefix
            if lowest_time is not None:
                data['Start Time'] = lowest_time
            data['Measures'] = n_images

    return data
