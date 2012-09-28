#!/usr/bin/env python
"""
The module handles log files relating to projects.
"""
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

import logging, uuid, os, copy

#
# GLOBALS
#

META_DATA = {'Start Time': 0, 'Prefix': 'unknown', 'Interval': 20.0, 
   'Description': 'Automatic placeholder description',
   'UUID': None, 'Measures': 0, 'Fixture': 'fixture_a',
   'Pinning Matrices': None, 'Manual Gridding': None}

IMAGE_ENTRY_KEYS = ['plate_1_area', 'grayscale_indices', 'grayscale_values',
    'plate_0_area', 'mark_X', 'mark_Y', 'plate_3_area', 'Time',
    'plate_2_area', 'File']

#
# FUNCTIONS
#


def get_image_dict(path, time, mark_X, mark_Y, grayscale_indices,
    grayscale_values, plate_areas):

    plate_str = "plate_{0}_area"

    image_entry = {'grayscale_indices': grayscale_indices,
        'grayscale_values': grayscale_values,
        'mark_X': mark_X, 'mark_Y': mark_Y,
        'Time': time, 'File': path}

    for i, plate in enumerate(plate_areas):

        image_entry[plate_str.format(i+1)] = plate

    return image_entry


def set_uuid(meta_data):

    meta_data['UUID'] = uuid.uuid1().get_urn().split(":")[-1]


def get_is_valid_meta_data(data):

    d_check = [k in data.keys() for k in META_DATA]

    return sum(d_check) > 3 

def get_is_valid_image_entry(data):

    d_check = [k in data.keys() for k in IMAGE_ENTRY_KEYS]

    return sum(d_check) > 6

def get_meta_data_from_file(path):

    try:

        fs = open(path, 'r')

    except:

        return None

    for l in fs:

        l = l.strip()
 
        if len(l) > 0:

            if l[0] == '{':

                l = eval(l)

                fs.close()
                if get_is_valid_meta_data(l):

                    return l

                else:

                    return None

    fs.close()
    return None


def get_meta_data(path = None):
    """This function will return a meta-data dict.
    If log-file doesn't exist, the meta-data will be generic.
    """

    meta_data = None

    if path is not None:
        meta_data = get_meta_data_from_file(path)

    if meta_data is None:

        meta_data = copy.copy(META_DATA)
        set_uuid(meta_data)

    return meta_data


def get_image_entries(path):

    try:

        fs = open(path, 'r')

    except:

        return None

    images = list()

    for line in fs:

        line = line.strip()

        if len(line) > 0:

            if line[0] == '{':

                line = eval(line)

                if get_is_valid_image_entry(line):

                    images.append(line)

    fs.close()

    return images


def get_log_file(path):

    return get_meta_data(path=path), get_image_entries(path)


def write_meta_data(path, meta_data=None, over_write=False):
    """Will write new data to file, over_write flag, if true
    will throw everything away that was there, else as much
    as possible will be kept."""

    if over_write:

        prev_meta_data = get_meta_data(None)
        prev_images = list()

    else:

        prev_meta_data, prev_images = get_log_file(path)

    if data is None:

        meta_data = prev_meta_data

    write_log_file(path, meta_data=meta_data, images=prev_images)


def write_log_file(path, meta_data=None, images=list()):
    """Creates a valid log file with valid meta-data row and
    writes all image-rows specified"""

    if meta_data is None:

        meta_data = get_meta_data(None)

    try:

        fs = open(path, 'w')

    except:

        return False

    images.insert(0, meta_data)
    images = ["{0}\n\r".format(l) for l in map(str, images)]
    fs.writelines(images)
    fs.close()
    return True


def get_image_dict_from_path(image_path):
    """
    f_settings = r_fixture.Fixture_Settings(
        script_path_root + os.sep + "config" + os.sep + "fixtures",
        fixture="fixture_a")
    """
    return None


def write_log_file_from_image_paths(log_file_path, image_paths,
        meta_data=None, only_append=False, force_over_write=False):
    """A list of image_paths are analysed and their info appended
    to the logfile.

    Only append will assume file is correct and add to the end.
    """

    new_images = list()

    for img in image_paths:

        new_img = get_image_dict_from_path(img)

        if new_img is not None:

            new_images.append(new_img)        

    if not only_append:

        if force_over_write:

            meta_data2 = get_meta_data(path=log_file_path)
            images = new_images

        else:

            meta_data2, images = get_log_file(log_file_path)
            images += new_images

        if meta_data is None:

            meta_data = meta_data2

            
        return write_log_file(log_file_path, meta_data, images)

    else:

        try:

            fs.open(log_file_path, 'a')

        except:

            return False

        images = ["{0}\n\r".format(l) for l in map(str, images)]
        fs.writelines(images)
        fs.close()

        return True


def append_image_dicts(path, images=list()):

    try:

        fs = open(path, 'a')

    except:

        return False

    images = ["{0}\n\r".format(l) for l in map(str, images)]
    fs.writelines(images)
    fs.close()

    return True
