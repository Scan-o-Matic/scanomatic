from __future__ import absolute_import

GRAYSCALES = {
    "Kodak": {
        "default": True,
        "lower_than_half_width": 350,
        "width": 55,
        "length": 28.3,
        "higher_than_half_width": 150,
        "min_width": 30,
        "sections": 23,
        "targets": [
            0, 2, 4, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46,
            50, 54, 58, 62, 66, 70, 74, 78, 82,
        ]
    },
    "SilverFast": {
        "default": False,
        "lower_than_half_width": 350,
        "width": 58,
        "length": 29.3,
        "higher_than_half_width": 150,
        "min_width": 30,
        "sections": 23,
        "targets": [
            -0.51755554588706332, 0.084950418163643349,
            2.0658933373355808, 4.8860848943085387,
            7.0895315818511051, 9.8090590380673746,
            12.460555769158788, 15.876292375724233,
            19.409915535646263, 23.969867902445415,
            27.9441242562445, 31.819161504807759,
            37.039832319927292, 41.121548316057002,
            45.627161530667792, 51.292846925654558,
            55.58899433057104, 59.280150768436272,
            64.618357884363064, 67.998085441870742,
            71.535859722426736, 75.139990306748999,
            76.5794216925988,
        ]
    }
}


def getGrayscales():

    return list(GRAYSCALES.keys())


def getDefualtGrayscale():

    for gs in getGrayscales():

        if GRAYSCALES.get("default", False):
            return gs


def getGrayscale(grayScaleName):

    if grayScaleName in getGrayscales():
        return GRAYSCALES[grayScaleName]
    else:
        raise Exception("{0} not among known grayscales {1}".format(
            grayScaleName, getGrayscales()))
