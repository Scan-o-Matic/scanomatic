"""Handler for writing/reading image data"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
import os
import glob

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.paths as paths
import scanomatic.io.logger as logger

#
# CLASSES
#


class Image_Data(object):

    _LOGGER = logger.Logger("Static Image Data Class")
    _PATHS = paths.Paths()

    @staticmethod
    def writeImage(path, imageIndex, features, nPlates, measure=None):

        path = os.path.join(Image_Data.path2dataPathTuple(
            path, imageIndex=imageIndex))

        if features is None:
            Image_Data._LOGGER.warning(
                "Image {0} had no data".format(imageIndex))
            return

        if measure is None:
            measure = ('blob', 'sum')

        plates = [None] * nPlates
        for pID in xrange(nPlates):
            if features[pID] is not None:

                lD1 = len(features)
                lD2 = len(features[0])
                plates[pID] = np.zeros((lD1, lD2)) * np.nan
                fPlate = features[pID]

                for id1 in xrange(lD1):
                    d1V = fPlate[id1]
                    for id2 in xrange(lD2):
                        cell = d1V[id2]
                        if cell is not None:
                            plates[pID][id1, id2] = cell[measure[0]][measure[1]]

        Image_Data._LOGGER.info("Saved Image Data '{0}'".format(path))
        np.save(path, plates)

    @staticmethod
    def iterWriteImageFromXML(path, xmlObject, measure=0):

        scans = xmlObject.get_scan_times().size
        nPlates = max(xmlObject.get_data().keys())
        data = xmlObject.get_data()

        for idS in range(scans):
            features = [None] * nPlates
            for idP in range(nPlates):
                features[idP] = data[idP][:, :, idS]

            Image_Data.writeImage(path, idS, features, nPlates, measure=measure)

    @staticmethod
    def writeTimes(path, imageIndex, imageMetaData, overwrite=False):

        if not overwrite:
            currentData = Image_Data.readTimes(path)
        else:
            currentData = np.array([], dtype=np.float)

        if not (imageIndex < currentData.size):
            currentData = np.r_[currentData, [None] * (1 + imageIndex -
                                                       currentData.size)]

        currentData[imageIndex] = imageMetaData['Time']
        np.save(Image_Data.path2dataPathTuple(path, times=True), currentData)

    @staticmethod
    def writeTimesFromXML(path, xmlObject):

        np.save(os.path.join(Image_Data.path2dataPathTuple(
            path, images=True)), xmlObject.get_scan_times())

    @staticmethod
    def readTimes(path):

        path = os.path.join(Image_Data.path2dataPathTuple(path, times=True))
        if os.path.isfile(path):
            return np.load(path)
        else:
            return np.array([], dtype=np.float)

    @staticmethod
    def readImage(path):

        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    @staticmethod
    def path2dataPathTuple(path, imageIndex="*", times=False):
        pathDir = os.path.dirname(path)
        pathBasename = os.path.basename(path)
        if (len(pathBasename) == 0):
            if times:
                pathBasename = Image_Data._PATHS.image_analysis_time_series
            else:
                pathBasename = Image_Data._PATHS.image_analysis_img_data

        return pathDir, pathBasename.format(imageIndex)

    @staticmethod
    def iterReadImages(path):

        for p in glob.iglob(os.path.join(Image_Data.path2dataPathTuple(path))):
            yield Image_Data.readImage(p)
