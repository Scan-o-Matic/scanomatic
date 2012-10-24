#
# INTERNAL DEPENDENCIES
#

from src.model_generic import *

#
# MODELS
#

model = {

'mode-selection-top-fixture': 'Fixture',
'mode-selection-top-poly': 'Cell Count Polynomial',

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

'fixture-select-radio-edit': 'Edit Existing Fixture:',
'fixture-select-radio-new': 'Create New Fixture:',
'fixture-select-column-header': 'Fixtures'
}


specific_fixture_model = {

'fixture': None,
'original-image-path': None
}
