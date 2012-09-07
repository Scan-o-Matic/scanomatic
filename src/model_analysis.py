import copy

#
# FUNCTIONS
#

def copy_model(model):

    tmp_model = dict()

    for k in model.keys():

        tmp_model[k] = model[k]

    return copy.deepcopy(tmp_model)


def copy_part_of_model(model, keys):

    tmp_model = dict()

    for k in keys:

        tmp_model[k] = model[k]

    return copy.deepcopy(model)


def link_to_part_of_model(model, keys):

    return Model_Link(model, keys)

#
# MODEL LINK
#

class Model_Link(object):

    def __init__(self, source_model, link_keys):

        self._source_model = source_model
        self._private_model = dict()
        self._link_keys = link_keys

    def __getattr__(self, key):

        if key in self._link_keys:

            return self._source_model[key]

        else:

            return self._private_model[key]

    def __setattr__(self, key, val):

        if key in self._link_keys:

            self._source_model[key] = val

        else:

            self._private_model[key] = val

    def keys(self):

        return link_keys + self._private_model.keys()

#
# MODELS
#

model = {
#ANALYSIS TOP ROOT
'analysis-top-root-project_button-text': 'Analyse Project',
'analysis-top-root-tpu_button-text': 'Manual Analyse Transparency Image',
'analysis-top-root-color_button-text': 'Manual Analyse Color Image',
'analysis-top-root_button-text': 'Select Analysis Mode',

#ANALYSIS STAGE ABOUT
'analysis-stage-about-text':
"""<span size="x-large"><u>Analysis Options</u></span>

<span weight="heavy">Project Analysis</span>
<span><i>This is for when you want to re-run the analysis of a project.
That is, projects normally, automatically starts an analysis themselves
as soon as they are done.</i></span>

<span weight="heavy">Manual Analyse Transparency Image</span>
<span><i>This is for manually selecting colonies of transparency
images and getting information about them. That is, for example,
quatative drop-tests or calibration experiments.</i></span>

<span weight="heavy">Manual Analyse Color Image</span>
<span><i>This is for analysing reflective color scan image --
to get quatitative values of average pigmentation.</i></span>
""",

#ANALYSIS TOP PROJECT
'analysis-top-project-start-text': 'Start',

#ANALYSIS STAGE PROJECT
'analysis-stage-project-title': 
"<span weight='heavy' size='large'>Analyse Project</span>",

'analysis-stage-project-file': 'Select Log File to Process',
'analysis-stage-project-log_file_button-text': 'Browse...',

'analysis-stage-project-output_folder': 
'Output Folder (relative log-file path)',

'analysis-stage-project-output_folder-ok': 'Nice choice!',

'analysis-stage-project-output_folder-warning': 
'That folder exists, might not be optimal to write into previous analysis',

'analysis-stage-project-plates': 'Tweak gridding',
'analysis-stage-project-keep_gridding': 'Keep grids as specified in log-file',
'analysis-stage-project-plate-label': 'Plate {0]',

#ANALYSIS PROJECT FILE
'analysis-project-log_file': '',
'analysis-project-log_file_dir':'',
'analysis-project-output-default': 'analysis',
'analysis-project-output-path': 'analysis',
'analysis-project-plates': 0,
'analysis-project-pinning-default': '1536',
'analysis-project-pinnings': (),
'analysis-project-pinnings-from-file': (),

#ANALYSIS TOP IMAGE SELECTION
'analysis-top-image-selection-next': 'Normalisation',

#ANALYSIS STAGE IMAGE SELECTION
'analysis-stage-image-selection-title':
"<span weight='heavy' size='large'>Image(s) Selection</span>",

'analysis-stage-image-selection-images': 'Images:',
'analysis-stage-image-selection-fixture': 'Images contain a fixture',
'analysis-stage-image-selection-dialog-button': 'Select Images...',
'analysis-stage-image-selection-file-dialogue-title': 'New Images to Analyse',

'analysis-stage-image-selection-file-filter': {'filter_name': 'Image Files',
'mime_and_patterns': (('IMAGE/TIFF', '*.[tT][iI][fF]'),
('IMAGE/TIF','*.[tT][iI][fF][fF]'))},

'analysis-stage-image-selection-list-column-title': 'Path',

#ANALYSIS STAGE IMAGE NORM MANUAL
'analysis-stage-image-norm-manual-title':
"<span weight='heavy' size='large'>Manual Image Normalisation</span>",

#GENERIC
'pinning-matrices': {'A: 8 x 12 (96)': (8, 12), 
                    'B: 16 x 24 (384)': (16, 24),
                    'C: 32 x 48 (1536)': (32, 48),
                    'D: 64 x 96 (6144)': (64, 96),
                    '--Empty--': None}

}

specific_transparency = {
'mode': 'transparency',
'fixture': True,
'images-list-model': None,
'image': -1
}
