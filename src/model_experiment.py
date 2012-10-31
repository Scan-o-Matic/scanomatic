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
}

specific_project_model = {
'fixture': None,
}
