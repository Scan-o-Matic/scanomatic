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

    m = Experiment_GUI_Model(private_model=model,
        generic_model=model_generic.get_model())

    return m


def copy_model(model):

    return model_generic.copy.deepcopy(model)

#
# SUBCLASSING
#

class Experiment_GUI_Model(model_generic.Model): pass

#
# MODELS
#

model = {

#ABOUT
##TOP
'mode-selection-top-project': 'Project',
'mode-selection-top-gray': 'One Image: Gray',
'mode-selection-top-color': 'One Image: Color',

##STAGE
'project-stage-about-text': """
<span size="x-large"><u>Experiment Options</u></span>

<span weight="heavy">Project</span>
<span><i>For scanning growth over a couple of days.
The standard, intended use of Scan-o-Matic.</i></span>

<span weight="heavy">One Image: Gray</span>
<span><i>This is for those that need to take only one image,
everything else being like the normal experiment runs.
E.g. for calibration experiment needed to be able to do the 
cell count transpose.</i></span>

<span weight="heavy">One Image: Color</span>
<span><i>This is for color images such as those used for
pigmentation analysis.</i></span>
""",

#PROJECT - SETUP
##TOP

##STAGE
###METADATA
'project-stage-meta': 'Meta-Data',
'project-stage-select_root':'Select Experiments Root Directory',
'project-stage-prefix': 'Project Prefix',
'project-stage-prefix-ok': 'Prefix is good',
'project-stage-prefix-warn': 'Duplcate prefix name or illegal name',
'project-stage-planning-id': 'Project ID from Planner',
'project-stage-desc': 'Project Description',
###FIXTURE AND SCANNER
'project-stage-fixture_scanner': 'Scanner and Fixture',
'project-stage-scanner-claim-fail': 'Failed to claim the scanner!\nSomeone is probably using it...',
'project-stage-scanner': 'Scanner:',
'project-stage-fixture': 'Fixture:',
###TIME SETTINGS
'project-stage-time_settings': 'Time Settings',
'project-stage-duration': 'Duration',
'project-stage-interval': 'Interval',
'project-stage-scans': 'Scans',
'project-stage-duration-ok': 'Great choice!',

'project-stage-duration-warn': 
'Invalid request, previous/default value will be used',

'project-stage-duration-format': "{0} days {1} hours {2} mins",
'project-stage-interval-format': "{0} mins",
'project-stage-scans-format': "{0}",

###PINNING
'project-stage-plates': 'Pinnings / Plate Formats',

#PROJECT - RUNNING
'project-running': """
<span weight="heavy">Project {0} is now running...</span>
<span><i>You may close this tab now.
If you want details about the progress, look under status.</i></span>
"""
}

specific_project_model = {
##PHYSICAL STUFF
'scanner': None,
'fixture': None,

##DATA
'experiments-root': '',
'experiment-prefix': None,
'experiment-desc': None,
'experiment-id': None,

##PLATES
'plate-areas': 0,
'pinnings-list': None,

##MARKERS
'marker-count': 0,
'ref-marker-positions': None,
'marker-path': None,

##GRAYSCALE
'grayscale': False,
'grayscale-area': None,

##DURATION
'duration-settings-order': ['duration', 'scans', 'interval'],
'interval': 20,
'scans': 217,
'duration': 72,
}
