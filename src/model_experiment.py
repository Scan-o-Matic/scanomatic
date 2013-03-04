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

def get_gui_model(paths=None):

    m = Experiment_GUI_Model(private_model=model,
        generic_model=model_generic.get_model(), paths=paths)

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
'project-stage-select_root':'Set Root Directory',
'project-stage-prefix': 'Project Prefix',
'project-stage-prefix-ok': 'Prefix is good',
'project-stage-prefix-warn': 'Duplcate prefix name or illegal name',
'project-stage-planning-id': 'Project & Scan Layout Tags',
'project-stage-desc': 'Project Description',
'project-stage-desc-suggestion': 'Plate 1 ""; Plate 2 ""; Plate 3 ""; Plate 4 ""',
'project-stage-view_of_fixture': 'View in Scanner',
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
""",

#ONE

##INTRO
'one-stage-intro-transparency': """
<span weight="heavy">Manually make transparency scan(s)</span>
<span><i>This will create a new folder using current date and time.
You will get a first pass analysis run automaticcaly if you select fixture.
(It can be done later under the analysis-tab).</i></span>
<span><i>If you select to 'Run Through' it will scan one image and then free the scanner.
If you select to just power on, you will lock turn the scanner on and it will
allow you to scan one or more images before you free it again.</i></span>
""",

'one-stage-intro-color': """
<span weight="heavy">Manually make reflective color scan(s)</span>
<span><i>This will create a new folder using current date and time.
You will get a first pass analysis run automaticcaly if you select fixture.
(It can be done later under the analysis-tab).</i></span>
<span><i>If you select to 'Run Through' it will scan one image and then free the scanner.
If you select to just power on, you will lock turn the scanner on and it will
allow you to scan one or more images before you free it again.</i></span>
""",

##FIXTURE AND SCANNER
'one-stage-fixture_scanner': 'Scanner & Fixture Selection',
'one-stage-scanner': 'Select Scanner',
'one-stage-fixture': 'Select Fixture (if any)',
'one-stage-no-fixture': "No fixture/don't run first pass analysis",

##RUN-TYPES
'one-stage-run-frame': 'Select Run-Mode/Stage',
'one-stage-power-up': 'Turn on Power',
'one-stage-run-all': 'Run Through It All',
'one-stage-scan': 'Scan',
'one-stage-complete': 'Complete',

##PROGRESS
'one-stage-progress': 'Progress',
'one-stage-progress-colors': 
'White (Not run yet), Blue (Running), Green (Completed), Red (Failed), Grey (Omitted)',

###PATTERNS
'one-stage-progress-not-run': 
"<span size='large' background='white' foreground='black'>{0}</span>",
'one-stage-progress-completed':
"<span size='large' background='green' foreground='black'>{0}</span>",
'one-stage-progress-running':
"<span size='large' background='blue' foreground='black'>{0}</span>",
'one-stage-progress-pass':
"<span size='large' background='grey' foreground='black'>{0}</span>",
'one-stage-progress-error':
"<span size='large' background='red' foreground='black'>{0}</span>",

###TEXTS
'one-stage-progress-power-on': ' Power On ',
'one-stage-progress-scan': ' Scan ',
'one-stage-progress-power-off': ' Power Off ',
'one-stage-progress-analysis': ' First Pass Analysis ',
'one-stage-progress-done': ' Done ',
}

specific_project_model = {
##PHYSICAL STUFF
'scanner': None,
'fixture': None,

##DATA
'experiments-root': '',
'experiment-prefix': None,
'experiment-desc': '',
'experiment-id': '',
'experiment-scan-layout-id': '',

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

specific_one_color_model = {
'type': 'color',
'scan-mode': 'COLOR',
'scanner': None,
'fixture': None,
'stage': None,
'image': 0
}

specific_one_transparency_model = {
'type': 'transparency',
'scan-mode': 'TPU',
'scanner': None,
'fixture': None,
'stage': None,
'image': 0,
}
