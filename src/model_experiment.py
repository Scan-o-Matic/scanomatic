#
# DEPENDENCIES
#

import numpy as np

#
# INTERNAL DEPENDENCIES
#

from src.model_generic import *

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
'project-stage-scanner': 'Scanner:',
'project-stage-fixture': 'Fixture:',
###TIME SETTINGS
'project-stage-time_settings': 'Time Settings',
'project-stage-duration': 'Duration',
'project-stage-interval': 'Interval',
'project-stage-scans': 'Scans',
###PINNING
'project-stage-plates': 'Pinnings / Plate Formats',

}

specific_project_model = {
'fixture': None,
'experiments-root': '',
}
