import ConfigParser
import numpy as np

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.paths as paths
from scanomatic.io.logger import Logger

#
# GLOBALS
#

_GRAYSCALE_PATH = paths.Paths().analysis_grayscales

_GRAYSCALE_CONFIGS = ConfigParser.ConfigParser()
_KEY_DEFUALT = 'default'
_KEY_TARGETS = 'targets'

GRAYSCALE_SCALABLE = ('width', 'min_width', 'lower_than_half_width',
                      'higher_than_half_width', 'length')

GRAYSCALE_CONFIG_KEYS = (_KEY_DEFUALT, _KEY_TARGETS,
                         'sections') + GRAYSCALE_SCALABLE

_GRAYSCALE_VALUE_TYPES = {
    'default': bool,
    'targets': eval,
    'sections': int,
    'width': float,
    'min_width': float,
    'lower_than_half_width': float,
    'higher_than_half_width': float,
    'length': float,
}

_logger = Logger("Grayscale settings")
#
# METHODS
#


def getGrayscales():

    try:
        _GRAYSCALE_CONFIGS.readfp(open(_GRAYSCALE_PATH, 'r'))
    except:
        _logger.critical(
            "Settings for grayscales not found at: " + _GRAYSCALE_PATH)
    return _GRAYSCALE_CONFIGS.sections()


def getDefualtGrayscale():

    for gs in getGrayscales():

        if _GRAYSCALE_CONFIGS.getboolean(gs, _KEY_DEFUALT):
            return gs


def getGrayscale(grayScaleName):

    if grayScaleName in getGrayscales():
        return {k: _GRAYSCALE_VALUE_TYPES[k](v) for k, v in
                _GRAYSCALE_CONFIGS.items(grayScaleName)}
    else:
        raise Exception("{0} not among known grayscales {1}".format(
            grayScaleName, getGrayscales()))


def getGrayscaleTargets(grayScaleName):

    targets = getGrayscale(grayScaleName)[_KEY_TARGETS]
    try:
        return _GRAYSCALE_VALUE_TYPES[_KEY_TARGETS](targets)
    except:
        return targets


def _saveConfig():

    with open(_GRAYSCALE_PATH, 'w') as configFile:
        _GRAYSCALE_CONFIGS.write(configFile)


def _setNewDefault(grayScaleName):

    for s in _GRAYSCALE_CONFIGS.sections():

        _GRAYSCALE_CONFIGS.set(s, _KEY_DEFUALT, str(s == grayScaleName))


def updateGrayscaleValues(grayScaleName, **kwargs):

    if grayScaleName in getGrayscales():

        for k, v in kwargs.items():

            if k in GRAYSCALE_CONFIG_KEYS:
                if k == _KEY_DEFUALT and v is True:
                    _setNewDefault(grayScaleName)
                else:
                    _GRAYSCALE_CONFIGS.set(grayScaleName, k, str(v))

        _saveConfig()

    else:

        raise Exception("{0} not among known grayscales {1}".format(
            grayScaleName, getGrayscales()))


def addGrayscale(grayScaleName, **kwargs):

    if grayScaleName in getGrayscales():

        raise Exception("{0} already exists!".format(grayScaleName))

    missingKeys = set(GRAYSCALE_CONFIG_KEYS).difference(kwargs.keys())

    if len(missingKeys) > 0:

        raise Exception("Missing {0} keys: {1}".format(
            len(missingKeys), tuple(missingKeys)))

    _GRAYSCALE_CONFIGS.add_section(grayScaleName)
    for k, v in kwargs.items():
        if k == _KEY_DEFUALT and v is True:
            _setNewDefault(grayScaleName)
        else:
            _GRAYSCALE_CONFIGS.set(grayScaleName, k, str(v))

    _saveConfig()


def validateFromData(name, source, target):

    if None in (source, target):
        return False

    if name not in getGrayscales():
        return False

    if list(target) != getGrayscaleTargets(name):
        return False

    if len(target) != len(source):
        return False

    #A true grayscale is monotoniously increasing or decreasing
    #Given that the fitted log2_curve is,
    #
    # y = a * x**3 + b * x**2 + c * x + d
    #
    #and thus its derivative,
    #
    # dy/dx = 3 * a * x ** 2 + 2 * b * x + c
    #
    #may not change sign for the range of interest. That is it must all
    #be positive or all negative.
    #

    #The polynomial coefficients are extracted
    polyCoefficients = np.polyfit(target, source, 3)
    #The derivative's coefficients calculated
    derivativeCoefficients = np.arange(4)[::-1] * polyCoefficients
    #A numpy polynomial created from the derivative coefficients
    derivativePolynomial = np.poly1d(derivativeCoefficients)
    #The sign of the derivative evaluated for the range of a 8bit image
    #4 evaluations per step. It would probably do with less
    derivativeValues = derivativePolynomial(np.linspace(0, 255, 1024))
    #For all non-zero values, the check the sign
    derivativeSigns = (derivativeValues[
        derivativeValues.nonzero()] > 0).astype(np.bool)
    #If either all are True OR if None are, the method returns True
    return derivativeSigns.all() or derivativeSigns.sum() == 0


def validate(fixture_settings):

    return validateFromData(
        fixture_settings['grayscaleName'],
        fixture_settings['grayscaleSource'],
        fixture_settings['grayscaleTarget'])
