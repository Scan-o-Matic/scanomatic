__author__ = 'martin'

import os
import glob
import re

import scanomatic.io.logger as logger
import scanomatic.io.paths as paths

_logger = logger.Logger("Legacy compatibility")


def patch_image_file_names_by_interval(path, interval=20.0):
    """

    :param path: Directory containing the images.
    :type path: str
    :param interval: Interval between images
    :type interval: float
    :return: None
    """

    pattern = re.compile(r"(.*)_\d{4}\.tiff")
    sanity_threshold = 3

    source_pattern = "{0}_{1}.tiff"
    target_pattern = paths.Paths().experiment_scan_image_pattern

    images = tuple(os.path.basename(i) for i in glob.glob(os.path.join(path, '*.tiff')))

    if not images:
        _logger.error("Directory does not contain any images")
        return

    base_name = ""
    included_images = 0
    for i in images:

        match = pattern.match(i)
        if match:
            included_images += 1
            if not base_name:
                base_name = match.groups()[0]
            elif match.groups()[0] != base_name:
                _logger.error("Conflicting image names, unsure if '{0}' or '{1}' is project name".format(
                    base_name, match.groups()[0]))
                return
        else:
            _logger.info("Skipping file '{0}' since it doesn't seem to belong in project".format(i))

    _logger.info("Will process {0} images".format(included_images))

    image_index = 0
    processed_images = 0
    index_length = 4

    while processed_images < included_images:

        source = os.path.join(path, source_pattern.format(base_name, str(image_index).zfill(index_length)))
        if os.path.isfile(source):
            os.rename(source, os.path.join(path, target_pattern.format(
                base_name, str(image_index).zfill(index_length), image_index * 60.0 * interval)))
            processed_images += 1
        else:
            _logger.warning("Missing file with index {0} ({1})".format(image_index, source))

        image_index += 1
        if image_index > included_images * sanity_threshold:
            _logger.error("Aborting becuase something seems to be amiss." +
                          " Currently attempting to process image {0}".format(image_index) +
                          " for a project which should only contain {0} images.".format(included_images) +
                          " So far only found {0} images...".format(processed_images))
            return

    _logger.info("Successfully renamed {0} images in project {1} using {2} minutes interval".format(
        processed_images, base_name, interval))