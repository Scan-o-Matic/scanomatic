#
# DEPENDENCIES
#

import numpy as np

#
# INTERNAL DEPENDENCIES
#

import src.model_generic as model_generic

#
# FUNCTIONS
#

def get_gui_model():

    m = Calibration_GUI_Model(private_model=model,
        generic_model=model_generic.get_model())

    return m


def copy_model(model):

    return model_generic.copy.deepcopy(model)

#
# SUBCLASSING
#

class Calibration_GUI_Model(model_generic.Model): pass

#
# MODELS
#

model = {

#ABOUT
##TOP
'mode-selection-top-fixture': 'Fixture',
'mode-selection-top-poly': 'Cell Count Polynomial',

##STAGE
'calibration-stage-about-text': """
<span size="x-large"><u>Calibration Options</u></span>

<span weight="heavy">Fixture</span>
<span><i>Fixture Calibration is done when a new fixture has been
constructed or to adjust an old fixture calibration. You will need
an image taken of the fixture in question.</i></span>

<span weight="heavy">Cell Count Polynomial</span>
<span><i>If a calibration experiment has been performed and the
results have been analysed, then here the csv-file can be used
to invoke a new calibration polynomial.</i></span>
""",

#FIXTURE SELECT
'fixture-select-title': """<span size="large">Fixture To Work With</span>""",

'fixture-select-radio-edit': 'Edit Existing Fixture:',
'fixture-select-radio-new': 'Create New Fixture:',
'fixture-select-column-header': 'Fixtures',
'fixture-select-new-name-duplicate': 'Illegal name, probably because it is a duplicate...',
'fixture-select-new-name-ok': 'Good choice!',
'fixture-select-next': 'Marker Calibration',

#FIXTURE MARKER CALIBRATION
'fixture-calibration-title': 
"""<span size="large">Image Selection and Marker Detection</span>""",
'fixture-calibration-next': 'Segmenting Fixture',
'fixture-calibration-select-im': 'Select Image',
'fixture-calibration-select-scan': 'Scan Fixture Now',
'fixture-calibration-marker-number': 'Number of calibration points',
'fixture-calibration-marker-detect': 'Run Detection',
'fixture-image-dialog': 'Select Image Using The Particular Fixture',
'fixture-image-file-filter': {'filter_name': 'Image Files',
'mime_and_patterns': (('IMAGE/TIFF', '*.[tT][iI][fF]'),
('IMAGE/TIF','*.[tT][iI][fF][fF]'))},
'scan-fixture-text': "Select scanner that you've placed the fixture in",
'scan-fixture-im-path': '',

#FIXTURE SEGMENTATION
'fixture-segmentation-title': """<span size="large">Segmenting Out Interests
</span>""",
'fixture-segmentation-gs': 'Fixture has a Kodak Grayscale',
'fixture-segmentation-next': 'Save the Fixture Calibration',
'fixture-segmentation-plates': 'Number of Plates',
'fixture-segmentation-column-header-segment': 'Segment',
'fixture-segmentation-column-header-ok': 'OK',
'fixture-segmentation-grayscale': 'Grayscale',
'fixture-segmentation-plate': 'Plate {0}',
'fixture-segmentation-nok': '',
'fixture-segmentation-ok': 'Yes',

#FIXTURE SEGMENTATION
'fixture-save-title': 
"""<span size="large">Fixture is saved and ready to use!</span>""",
}


specific_fixture_model = {

'fixture': None,
'fixutre-file': None,
'new_fixture': False,

'original-image-path': None,
'im': None,
'im-path': None,
'im-original-scale': 1.0,
'im-scale': 1.0,
'im-not-loaded': 'No image selected',

'grayscale-exists': True,
'grayscale-image-text': 'G',
'grayscale-coords': list(),
'grayscale-targets': np.asarray([82, 78, 74, 70, 66, 62, 58, 54, 50, 46,
                            42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 4, 2, 0]),
'grayscale-sources': None,

'plate-coords': [None, None, None, None],

'markers': 0,
'marker-path': None,
'marker-positions': list(),

'active-segment': None,
'active-source': None,
'active-target': None,
}
