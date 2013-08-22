"""Resource module for handling the greyscales."""

__author__ = "Martin Zackrisson, Andreas Skyman"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.994"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import ConfigParser

#
# INTERNAL DEPENDENCIES
#

import resource_path

#
# GLOBALS
#

_GRAYSCALE_PATH = resource_path.Paths().analysis_graycsales

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

#
# METHODS
#


def getGrayscales():

    try:
        _GRAYSCALE_CONFIGS.readfp(open(_GRAYSCALE_PATH, 'r'))
    except:
        pass
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
