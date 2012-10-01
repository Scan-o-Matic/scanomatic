import copy
import os

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
'analysis-stage-project-file-prefix': 'Prefix',
'analysis-stage-project-file-desc': 'Description',
'analysis-stage-project-file-images': 'Images',

'analysis-stage-project-output_folder': 
'Output Folder (relative log-file path)',

'analysis-stage-project-output_folder-ok': 'Nice choice!',

'analysis-stage-project-output_folder-warning': 
'That folder exists, might not be optimal to write into previous analysis',

'analysis-stage-project-plates': 'Tweak gridding',
'analysis-stage-project-keep_gridding': 'Keep grids as specified in log-file',
'analysis-stage-project-plate-label': 'Plate {0}',
'analysis-stage-project-select-log-file-dialog': 'Select Log File',

'analysis-stage-project-select-log-file-filter': {'filter_name': 'Log Files',
'mime_and_patterns': (('TEXT', '*.log'),)},

#ANALYSIS TOP IMAGE SELECTION
'analysis-top-image-selection-next': 'Normalisation',

#ANALYSIS STAGE IMAGE SELECTION
'analysis-stage-image-selection-title':
"<span weight='heavy' size='large'>Image(s) Selection</span>",

'analysis-stage-image-selection-fixture': 'Images contain a fixture',
'analysis-stage-image-selection-dialog-button': 'Select Images...',
'analysis-stage-image-selection-file-dialogue-title': 'New Images to Analyse',

'analysis-stage-image-selection-file-filter': {'filter_name': 'Image Files',
'mime_and_patterns': (('IMAGE/TIFF', '*.[tT][iI][fF]'),
('IMAGE/TIF','*.[tT][iI][fF][fF]'))},

'analysis-stage-image-selection-list-column-title': 'Image Path(s)',

'analysis-stage-image-selection-continue-log': 'Continue previous log session',
'analysis-stage-image-selection-continue-button': 'Select CSV-file...',
'analysis-stage-image-selection-logging-title': 'Logging interests',
'analysis-stage-image-selection-compartments': 'Compartments',
'analysis-stage-image-selection-measures': 'Measures',

#ANALYSIS TOP IMAGE NORMALISATION
'analysis-top-image-normalisation-next': 'Plate Sectioning',

#ANALYSIS STAGE IMAGE NORM MANUAL
'analysis-stage-image-norm-manual-title':
"<span weight='heavy' size='large'>Manual Image Normalisation</span>",

'analysis-stage-image-norm-manual-measures': 'Measures',

#ANALYSIS TOP IMAGE SECTIONING
'analysis-top-image-sectioning-next': 'Plate Analysis',

#ANALYSIS STAGE IMAGE NORM MANUAL
'analysis-stage-image-sectioning-title':
"<span weight='heavy' size='large'>Marking out Plates</span>",

'analysis-stage-image-sectioning-help_text':
"""<i>If you have the same stuff in several images, mark out the plates
in the same order and watch the magic!</i>""",

#ANALYSIS TOP IMAGE SECTIONING
'analysis-top-image-plate-next_plate': 'Next Plate',
'analysis-top-image-plate-next_image': 'Next Image',
'analysis-top-image-plate-next_done': 'Done!',

#ANALYSIS STAGE IMAGE PLATE
'analysis-stage-image-plate-title':
"<span weight='heavy' size='large'>Analyis of Plate {0}</span>",

'analysis-stage-image-plate-name': 'Plate Description/Media:',
'analysis-stage-image-plate-lock_selection': 'Lock Selection Size',
'analysis-stage-image-plate-selection-width': 'Selection Width:',
'analysis-stage-image-plate-selection-height': 'Selection Height:',
'analysis-stage-image-plate-colony-name': 'Name specific experiment:',
'analysis-stage-image-plate-log-button': 'Record data for this experiment',

#ANALYSIS STAGE LOG
'analysis-stage-log-title': 'Log Book',
'analysis-stage-log-save': 'Save...',
'analysis-stage-log-save-dialog': 'Name your file:',

'analysis-stage-log-save-file-filter': {'filter_name': 'CSV Files',
'mime_and_patterns': (('TEXT/CSV', '*.[cC][sS][vV]'),)},

'analysis-stage-log-overwrite': 'The file "{0}" already exists, are you sure you want to overwrite it?',
'analysis-stage-log-saved': 'Your file has been saved!',
'analysis-stage-log-not-saved': "Your file wasn't saved",

#ANALYSIS TOP AUTO NORMALISATION AND SECTIONING
'analysis-top-auto-norm-and-section-next': 'Plate 1',

#ANALYSIS STAGE AUTO NORMALISATION AND SECTIONING
'analysis-stage-auto-norm-and-section-title':
"<span weight='heavy' size='large'>Automatic Normalisation and Sectioning</span>",

'analysis-stage-auto-norm-and-section-file': 'Use already detected',
'analysis-stage-auto-norm-and-section-fixture': 'Detect:',
'analysis-stage-auto-norm-and-section-run': 'Run it!',

#GENERIC
'pinning-matrices': {'A: 8 x 12 (96)': (8, 12), 
                    'B: 16 x 24 (384)': (16, 24),
                    'C: 32 x 48 (1536)': (32, 48),
                    'D: 64 x 96 (6144)': (64, 96),
                    '--Empty--': None},

'pinning-matrices-reversed': {(8, 12): 'A: 8 x 12 (96)', 
                    (16, 24): 'B: 16 x 24 (384)',
                    (32, 48): 'C: 32 x 48 (1536)',
                    (64, 96): 'D: 64 x 96 (6144)',
                    None: '--Empty--'},

'fixtures': list(),
'fixtures-path': 'src{0}config{0}fixtures{0}'.format(os.sep),
'not-implemented': "That feature hasn't been implemented yet!"

}

specific_project = {
'analysis-project-log_file': '',
'analysis-project-log_file_dir':'',
'analysis-project-output-default': 'analysis',
'analysis-project-output-path': 'analysis',
'analysis-project-plates': 0,
'analysis-project-pinning-default': '1536',
'analysis-project-pinnings': (),
'analysis-project-pinnings-from-file': (),
'analysis-project-pinnings-active': 'file'
}

specific_transparency = {
'mode': 'transparency',
'fixture': True,
'fixture-name': None,
'stage': None,
'manual-calibration-positions': None,
'manual-calibration-values': None,
'auto-transpose': dict(),
'images-list-model': None,
'image': -1,
'plate': -1,
'plate-coords': list(),
'plate-is-normed': False,
'plate-im-array': None,
'plate-section-im-array': None,
'plate-section-grid-cell': None,
'plate-section-features': None,
'plate-selection': {'u': None, 'l': None, 'w': None, 'h': None},
'lock-selection': None,
'selection-move-source': None,
'selection-origin': None,
'selection-size': None,
'selection-drawing': False,
'image-array': None,
'log-previous-file': None,
'log-interests': [None, None],
'log-meta-features': ('image','plate-coords','plate-index','plate-media',
'selected-plate-area', 'strain'),
'log-compartments-default': ('blob','background','cell'),
'log-measures-default': ('pixelsum', 'area', 'mean', 'median', 'IQR',
'IQR_mean', 'centroid', 'perimeter')
}

specific_log_book = {
'manual-calibration': None,
'log': list(),
'log-interests': None,
'images': list(),
'plate-names': list(),
'current-strain': None,
'measures': list()
}
