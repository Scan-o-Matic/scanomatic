#!/usr/bin/env python
"""
The module handles log files relating to projects.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.999"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import uuid
import os
import copy
import logging

#
# EXCEPTIONS
#


class Unknown_Meta_Data_Key(Exception):
    pass

#
# GLOBALS
#

META_DATA = {
    'Start Time': 0, 'Prefix': 'unknown', 'Interval': 20.0,
    'Description': 'Automatic placeholder description',
    'Version': __version__,
    'UUID': None, 'Measures': 0, 'Fixture': '', 'Scanner': '',
    'Pinning Matrices': None, 'Manual Gridding': None, 'Project ID': '',
    'Scanner Layout ID': ''}

IMAGE_ENTRY_KEYS = [
    'plate_1_area', 'grayscale_indices', 'grayscale_values',
    'plate_0_area', 'mark_X', 'mark_Y', 'plate_3_area', 'Time',
    'plate_2_area', 'File', 'Image Shape']

_logger = logging.getLogger("Project Log")
#
# FUNCTIONS
#


def get_meta_data_dict(**kwargs):

    md = copy.deepcopy(META_DATA)
    for k in kwargs:
        if k in md:
            md[k] = kwargs[k]
        else:
            raise Unknown_Meta_Data_Key(k)

    return md


def get_image_dict(path, time, mark_X, mark_Y, grayscale_indices,
                   grayscale_values, scale, plate_areas=None, img_dict=None,
                   image_shape=None):

    plate_str = "plate_{0}_area"

    image_entry = {'grayscale_indices': grayscale_indices,
                   'grayscale_values': grayscale_values,
                   'mark_X': mark_X, 'mark_Y': mark_Y,
                   'Time': time, 'File': path, 'Image Shape': image_shape,
                   'Scale': scale}

    if plate_areas is not None:
        for i, plate in enumerate(plate_areas):

            image_entry[plate_str.format(i + 1)] = plate

    elif img_dict is not None:

        i = 0
        while True:

            try:
                image_entry[plate_str.format(i + 1)] = img_dict[
                    plate_str.format(i)]
            except:
                break

            i += 1

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


def get_meta_data(path=None):
    """This function will return a meta-data dict.
    If log-file doesn't exist, the meta-data will be generic.
    """

    meta_data = None

    if path is not None:
        meta_data = get_meta_data_from_file(path)

    if meta_data is None:

        meta_data = copy.copy(META_DATA)
        set_uuid(meta_data)

    if ('Pinning Matrices' in meta_data and
            meta_data['Pinning Matrices'] is not None):

        for i, m in enumerate(meta_data['Pinning Matrices']):
            if isinstance(m, list):
                meta_data['Pinning Matrices'][i] = tuple(m)

    if 'Version' in meta_data:

        try:
            meta_data['Version'] = float(meta_data['Version'])
        except:
            pass

    return meta_data


def get_image_entries(path):

    _logger.info("Reading Image Entries {0}".format(path))

    try:

        fs = open(path, 'r')

    except:
        _logger.critical("Could not load 1-pass file '{0}'".format(path))

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


def get_image_from_log_file(path, image):

    meta_data, images = get_log_file(path)

    res = [im for im in images if image in im['File']]

    if len(res) > 0:

        return res[0]

    return None


def get_number_of_plates(path=None, meta_data=None, images=None):

    plates = -1

    if path is not None:

        meta_data, images = get_log_file(path)

    if meta_data is not None:

        if 'Pinning Matrices' in meta_data:

            plates = len(meta_data['Pinning Matrices'])

    if plates < 0 and images is not None:

        image = images[0]

        p_str = "plate_{0}_area"
        i = 0

        while p_str.format(i) in image:

            i += 1

        if i > 0:

            plates = i

    return plates


def write_meta_data(path, meta_data=None, over_write=False):
    """Will write new data to file, over_write flag, if true
    will throw everything away that was there, else as much
    as possible will be kept."""

    if over_write:

        prev_meta_data = get_meta_data(None)
        prev_images = list()

    else:

        prev_meta_data, prev_images = get_log_file(path)

    if meta_data is None:

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


def write_to_log_meta(path, partial_meta_data):

    meta_data, images = get_log_file(path)

    for k in partial_meta_data:

        meta_data[k] = partial_meta_data[k]

    write_log_file(path, meta_data=meta_data, images=images)


def _approve_image_dicts(images):

    for i_dict in images:

        if 'mark_X' in i_dict:

            i_dict['mark_X'] = list(i_dict['mark_X'])

        if 'mark_Y' in i_dict:

            i_dict['mark_Y'] = list(i_dict['mark_Y'])

        if ('grayscale_indices' in i_dict and
                i_dict['grayscale_indices'] is not None):

            i_dict['grayscale_indices'] = list(i_dict['grayscale_indices'])

        else:

            i_dict['grayscale_indices'] = None

        if ('grayscale_values' in i_dict and
                i_dict['grayscale_values'] is not None):

            i_dict['grayscale_values'] = list(i_dict['grayscale_values'])

        else:
            i_dict['grayscale_values'] = None

        if 'Time' not in i_dict and 'File' in i_dict:

            try:

                i_dict['Time'] = os.stat(i_dict['File']).st_mtime

            except:

                pass


def append_image_dicts(path, images=list()):

    _approve_image_dicts(images)

    try:

        fs = open(path, 'a')

    except:

        return False

    images = ["{0}\n\r".format(l) for l in map(str, images)]
    fs.writelines(images)
    fs.close()

    return True
