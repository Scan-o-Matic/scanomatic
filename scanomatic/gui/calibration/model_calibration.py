#
# DEPENDENCIES
#

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.model_generic as model_generic

#
# FUNCTIONS
#


def get_gui_model():

    m = Calibration_GUI_Model(
        private_model=model,
        generic_model=model_generic.get_model())

    return m


def copy_model(model):

    return model_generic.copy.deepcopy(model)

#
# SUBCLASSING
#


class Calibration_GUI_Model(model_generic.Model):
    pass

#
# MODELS
#

model = {

    'from': 'From:',
    'to': 'To:',
    'save': 'Save',

#ABOUT
##TOP
'mode-selection-top-fixture': 'Fixture',
'mode-selection-top-grayscale': 'Grayscale',
'mode-selection-top-poly': 'Cell Count Polynomial',

##STAGE
'calibration-stage-about-text': """
<span size="x-large"><u>Calibration Options</u></span>

<span weight="heavy">Fixture</span>
<span><i>Fixture Calibration is done when a new fixture has been
constructed or to adjust an old fixture calibration. You will need
an image taken of the fixture in question.</i></span>

<span weight="heavy">Grayscale</span>
<span><i>Grayscale calibration allows for setting up new models
of grayscales by comparing them to previously known grayscales.
To use this function, an image with both scales in it is required.</i></span>

<span weight="heavy">Cell Count Polynomial</span>
<span><i>If a calibration experiment has been performed and the
results have been analysed, then here the csv-file can be used
to invoke a new calibration polynomial.</i></span>
""",

#GRAYSCALE
    'grayscale-title': """<span size="large">Grayscale Calibration</span>""",
    'grayscale-load-image': """Select an Image with both grayscales""",
    'grayscale-new-model': """Add New Model""",
    'grayscale-mark-instructions': """Select the area of the""",
    'grayscale-source': 'Known reference',
    'grayscale-target': 'New scale',
    'grayscale-types': ['target', 'source'],
    'grayscale-colors': ['#A45656', '#56A456'],
    'grayscale-info-default':
    """First select an image, then mark the range of the reference grayscale.
After that, mark the grayscale you intend to calibrate. Both are properly
detected (pixel values are monotonious increasing or decreasing), you get
the option of commiting the new target values of the grayscale being
calibrated (click 'Save').""",
    'grayscale-info-done': """Calibration completed and new settings can now be
saved.""",
    'grayscale-info-saved': """New calibration target values have been saved
and are ready to be used.""",

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


specific_grayscale_model = {
    'active-type': 'source',
    'active-lineWidth': 1,
    'active-color': None,
    'active-alpha': 0.5,
    'active-source': None,
    'active-target': None,
    'active-changing': False,

    'source-sourceValues': None,
    'source-targetValues': None,
    'source-name': None,

    'target-sourceValues': None,
    'target-targetValues': None,
    'target-name': None,

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
'grayscale-type': None,
'grayscale-image-text': 'G',
'grayscale-coords': list(),
'grayscale-targets': None,
'grayscale-sources': None,

'plate-coords': [None, None, None, None],

'markers': 0,
'marker-path': None,
'marker-positions': list(),

'active-segment': None,
'active-source': None,
'active-target': None,
}
