import numpy as np
import os
import glob
import re

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
from scanomatic.io.pickler import unpickle_with_unpickler
#
#
#

_SECONDS_PER_HOUR = 60.0 * 60.0

#
# CLASSES
#


class ImageData(object):

    _LOGGER = logger.Logger("Static Image Data Class")
    _PATHS = paths.Paths()

    @staticmethod
    def write_image(analysis_model, image_model, features):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        return ImageData._write_image(analysis_model.output_directory, image_model.image.index, features,
                                      analysis_model.image_data_output_item,
                                      analysis_model.image_data_output_measure)

    @staticmethod
    def _write_image(path, image_index, features, output_item, output_value):

        path = os.path.join(*ImageData.directory_path_to_data_path_tuple(
            path, image_index=image_index))

        if features is None:
            ImageData._LOGGER.warning("Image {0} had no data".format(image_index))
            return

        number_of_plates = features.shape[0]
        plates = [None] * number_of_plates
        ImageData._LOGGER.info("Writing features for {0} plates ({1})".format(number_of_plates, features.shape))

        for plate_features in features.data:

            if plate_features is None:
                continue

            plate = np.zeros(plate_features.shape) * np.nan
            ImageData._LOGGER.info("Writing plate features for plates index {0}".format(plate_features.index))
            plates[plate_features.index] = plate

            for cell_features in plate_features.data:

                if output_item in cell_features.data:

                    compartment_features = cell_features.data[output_item]

                    if output_value in compartment_features.data:

                        try:
                            plate[cell_features.index[::-1]] = compartment_features.data[output_value]
                        except IndexError:

                            ImageData._LOGGER.critical(
                                "Shape mismatch between plate {0} and colony position {1}".format(
                                    plate_features.shape, cell_features.index))

                            return False
                    else:
                        ImageData._LOGGER.info("Missing data for colony position {0}, plate {1}".format(
                            cell_features.index,
                            plate_features.index))
                else:
                    ImageData._LOGGER.info("Missing compartment for colony position {0}, plate {1}".format(
                        cell_features.index,
                        plate_features.index
                    ))

        ImageData._LOGGER.info("Saved Image Data '{0}' with {1} plates".format(
            path, len(plates)))

        np.save(path, plates)
        return True

    @staticmethod
    def write_times(analysis_model, image_model, overwrite):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        global _SECONDS_PER_HOUR

        if not overwrite:
            current_data = ImageData.read_times(analysis_model.output_directory)
        else:
            current_data = np.array([], dtype=np.float)

        if not (image_model.image.index < current_data.size):
            current_data = np.r_[
                current_data,
                [None] * (1 + image_model.image.index - current_data.size)].astype(np.float)

        current_data[image_model.image.index] = image_model.image.time_stamp / _SECONDS_PER_HOUR
        np.save(os.path.join(*ImageData.directory_path_to_data_path_tuple(analysis_model.output_directory, times=True)),
                current_data)

    @staticmethod
    def read_times(path):

        path = os.path.join(*ImageData.directory_path_to_data_path_tuple(path, times=True))
        ImageData._LOGGER.info("Reading times from {0}".format(
            path))
        if os.path.isfile(path):
            return unpickle_with_unpickler(np.load, path)
        else:
            ImageData._LOGGER.warning("Times data file not found")
            return np.array([], dtype=np.float)

    @staticmethod
    def read_image(path):

        if os.path.isfile(path):
            return unpickle_with_unpickler(np.load, path)
        else:
            return None

    @staticmethod
    def directory_path_to_data_path_tuple(directory_path, image_index="*", times=False):

        if os.path.isdir(directory_path) and not directory_path.endswith(os.path.sep):
            directory_path += os.path.sep

        path_dir = os.path.dirname(directory_path)

        if times:
            path_basename = ImageData._PATHS.image_analysis_time_series
        else:
            path_basename = ImageData._PATHS.image_analysis_img_data

        return path_dir, path_basename.format(image_index)

    @staticmethod
    def iter_image_paths(path_pattern):

        return (p for p in glob.iglob(os.path.join(
            *ImageData.directory_path_to_data_path_tuple(path_pattern))))

    @staticmethod
    def iter_read_images(path):
        """A generator for reading image data given a directory path.

        Args:

            path (str): The path to the directory where the image data is

        Returns:

            Generator of image data

        Simple Usage::

            ``D = np.array((list(Image_Data.iterReadImages("."))))``

        Note::

            This method will not return image data as expected by downstream
            feature extraction.

        Note::

            This method does **not** return the data in time order.

        """
        for p in ImageData.iter_image_paths(path):
            yield ImageData.read_image(p)

    @staticmethod
    def convert_per_time_to_per_plate(data):
        """Conversion method for data per time (scan) as generated by the
        image analysis to data per plate as used by feature extraction,
        quality control and user output.

        Args:

            data (iterable):    A numpy array or similar holding data
                                per time/scan. The elements in data must
                                be numpy arrays.
        Returns:

            numpy array.    The data restrucutred to be sorted by plates.

        """
        if not hasattr(data, "shape"):
            data = np.array(data)

        try:
            new_data = [(None if data[0][plate_index] is None else [])
                        for plate_index in range(max(scan.shape[0] for scan in data if isinstance(scan, np.ndarray)))]
        except ValueError:
            ImageData._LOGGER.error("There is no data")
            return None

        for scan in data:
            for plate_id, plate in enumerate(scan):

                if plate is None:
                    continue

                new_data[plate_id].append(plate)

        for plate_id, plate in enumerate(new_data):

            if plate is None:
                continue

            p = np.array(plate)
            new_data[plate_id] = np.lib.stride_tricks.as_strided(
                p, (p.shape[1], p.shape[2], p.shape[0]),
                (p.strides[1], p.strides[2], p.strides[0]))

        return np.array(new_data)

    @staticmethod
    def read_image_data_and_time(path):
        """Reads all images data files in a directory and report the
        indices used and data restructured per plate.

        Args:

            path (string):  The path to the directory with the files.

        Retruns:

            tuple (numpy array of time points, numpy array of data)

        """
        times = ImageData.read_times(path)

        data = []
        time_indices = []
        for p in ImageData.iter_image_paths(path):

            try:
                time_indices.append(int(re.findall(r"\d+", p)[-1]))
                data.append(unpickle_with_unpickler(np.load, p))
            except AttributeError:
                ImageData._LOGGER.warning(
                    "File '{0}' has no index number in it, need that!".format(
                        p))

        try:
            times = np.array(times[time_indices])
        except IndexError:
            ImageData._LOGGER.error(
                "Could not filter image times to match data")
            return None, None

        sort_list = np.array(time_indices).argsort()
        return times[sort_list],  ImageData.convert_per_time_to_per_plate(
            np.array(data)[sort_list])
