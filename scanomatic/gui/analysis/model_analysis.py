#
# DEPENDENCIES
#

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.model_generic as model_generic
import scanomatic.io.paths as paths_module

#
# FUNCTIONS
#


def get_gui_model():

    m = Analysis_GUI_Model(
        private_model=model,
        generic_model=model_generic.get_model())

    return m


def copy_model(model):

    return model_generic.copy.deepcopy(model)

#
# SUBCLASSING
#


class Analysis_GUI_Model(model_generic.Model):
    pass

#
# GLOBALS
#

PATHS = paths_module.Paths()

#
# MODELS
#

model = {
#ANALYSIS TOP ROOT
'analysis-top-root-project_button-text': 'Project',
'analysis-top-root-tpu_button-text': 'Transparency Image',
'analysis-top-root-color_button-text': 'Color Image',
'analysis-top-root-1st_pass-text': 'Make Project',
'analysis-top-root-inspect-text': 'Inspect Analysis',
'analysis-top-root_button-text': 'Select Analysis Mode',
'analysis-top-root-features': 'Extract Features',
'analysis-top-root-convert': 'Convert Analysis',

#ANALYSIS STAGE ABOUT
'analysis-stage-about-text':
"""<span size="x-large"><u>Analysis Options</u></span>

<span weight="heavy">Project</span>
<span><i>This is for when you want to re-run the analysis of a project. That is, projects normally, automatically starts an analysis themselves
as soon as they are done.</i></span>

<span weight="heavy">Make Project</span>
<span><i>If you have to remake the first pass analysis so that you get a file that can be used for analysing a project, or
if you have a couple of images that you want to analyse as if they had been acquired by running an experiment.</i></span>

<span weight="heavy">Inspect Analysis</span>
<span><i>Inspect how griding worked to improve future griddings (and one day, maybe to manual adjustments to it)</i></span>

<span weight="heavy">Convert XML to native</span>
<span><i>If your analysis folder lacks the numpy-files associated with image analysis, then use this to convert the analysis slimmed xml-file
into the npy-files needed for feature extraction and quality control</i></span>

<span weight="heavy">Extract Features</span>
<span><i>Extracts growth parameters from an image analysis directory.</i></span>

<span weight="heavy">Transparency Image</span>
<span><i>This is for manually selecting colonies of transparency images and getting information about them. That is, for example,
quatative drop-tests or calibration experiments.</i></span>
""",

#ANALYSIS TOP INSPECT
'analysis-top-inspect-text': 'Select an analysis folder to get started',

#ANALYSIS STAGE INSPECT
'analysis-stage-inspect-not_selected': '--No Project Selected--',
'analysis-stage-inspect-select_button': 'Select Analysis To Inspect',
'analysis-stage-inspect-analysis-popup': "Select an analysis.run file in an analysis subfolder to your experiment",
'analysis-stage-inspect-file-filter': {'filter_name': 'Analysis Log Files',
'mime_and_patterns': (('TEXT', 'analysis.run'),)},
'analysis-stage-inspect-warning': 'Not all plates seem to have been gridded!',
'analysis-stage-inspect-plate-title': 'Plate {0} "{1}"',
'analysis-stage-inspect-plate-bad': 'This is a bad grid!',
'analysis-stage-inspect-plate-yn': 'This will cause the pinning of plate {0} to be removed. Do you wish to continue?',
'analysis-stage-inspect-plate-gone': 'Plate was removed',
'analysis-stage-inspect-plate-nohistory': 'Not in history',
'analysis-stage-inspect-plate-remove-warn': 'Failed to remove the selected plate from fixture history',
'analysis-stage-inspect-plate-drawing': 'Layout drawing as seen in:',
'analysis-stage-inspect-upload-button': 'Upload files to Precog-computer',
'analysis-stage-inspect-upload-install': 'Filezilla is lacking on your system, do you want to install it?',
'analysis-stage-inspect-upload-error': "Could not install.\n\nEither you didn't get the password right or decided not to install.\nOr this is not a debian-linux with filezilla in the repo!",
'analysis-stage-inspect-upload-gksu': "Please write password for administrative priviledges to install Filezilla",
'analysis-stage-inspect-error':
    """<span weight='heavy' size='large'>Error, no analysis done</span>""",

#ANALYSIS TOP PROJECT
'analysis-top-project-start-text': 'Start',

#ANALYSIS STAGE PROJECT
'analysis-stage-project-title':
"<span weight='heavy' size='large'>Analyse Project</span>",

'analysis-stage-project-file': 'Select Log File to Process',
'analysis-stage-project-log_file_button-text': 'Browse...',
'analysis-stage-project-file-invalid': 'No(t) valid log file loaded!',
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
'analysis-stage-project-select-log-file-dialog': 'Select Log File',

'analysis-stage-project-select-log-file-filter': {'filter_name': 'Log Files',
'mime_and_patterns': (('TEXT', '*.analysis'),)},
'analysis-stage-project-running-info':
"""<span weight="heavy">Running Analysis on '{0}'</span>
<span><i>This process runs in the background and will appear as any
experiment under running experiments.</i></span>""",
#ANALYSIS STAGE FIRST PASS
'analysis-stage-first-title':
"""<span weight='heavy' size='large'>First Pass Analysis</span>

<span><i>Add and order images in the order you consider chronological</i></span>""",
'analysis-stage-first-dir': 'Select directory for your project',
'analysis-stage-first-dir-title': 'Project Directory:',
'analysis-stage-first-file': 'Output File:',
'analysis-stage-first-where': 'Where',
'analysis-stage-first-local_fixture': 'Use fixture config copy in this directory',
'analysis-stage-first-column-title': 'Files in the order they will be analysed',
'analysis-stage-first-meta': 'Metadata',
'analysis-stage-first-meta-prefix': 'Prefix',
'analysis-stage-first-meta-id': 'Tags from planner',
'analysis-stage-first-meta-desc': 'Description',
'analysis-stage-first-fixture_scanner': 'Scanner & Fixture',
'analysis-stage-first-scanner': 'Scanner',
'analysis-stage-first-fixture': 'Fixture',
'analysis-stage-first-plates': 'Plates',
'analysis-stage-first-plates-number': 'Number of Plates:',
'analysis-stage-first-running-intro':
"""<span weight='heavy' size='large'>Building project '{0}'</span>
<span><i>Please allow the process to run to complete before you start
analysis of the project.</i></span>""",
'analysis-stage-first-running-working': 'Working...',
'analysis-stage-first-running-complete': 'Complete!',
'analysis-stage-first-running-error-path': "Could not create file, invalid path: '{0}'",
'analysis-stage-first-running-error-access': "Could not append image data to file," +
    " someone stole write permit to '{0}'",
'analysis-stage-first-running-error-img': "Image '{0}' failed to analyse .. skipping!\n",
'analysis-stage-first-id-warn': "Id-tags don't match with control number, this will cause import problems in PALM",

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
                      ('IMAGE/TIF', '*.[tT][iI][fF][fF]'))},

'analysis-stage-image-selection-list-column-title': 'Image Path(s)',

'analysis-stage-image-selection-continue-log': 'Continue previous log session',
'analysis-stage-image-selection-continue-button': 'Select CSV-file...',
'analysis-stage-image-selection-logging-title': 'Logging interests',
'analysis-stage-image-selection-compartments': 'Compartments',
'analysis-stage-image-selection-measures': 'Measures',
'analysis-stage-image-selection-calibration': 'Only calibration vector',

#ANALYSIS TOP IMAGE NORMALISATION
'analysis-top-image-normalisation-next': 'Plate Sectioning',

#ANALYSIS STAGE IMAGE NORM MANUAL
'analysis-stage-image-norm-manual-title':
"<span weight='heavy' size='large'>Manual Image Normalisation</span>",

'analysis-stage-image-norm-manual-measures': 'Measure Source',
'analysis-stage-image-norm-manual-targets': 'Measure Target',

'analysis-stage-image-norm-manual-useGrayscale': 'Use Grayscale',

#ANALYSIS TOP IMAGE SECTIONING
'analysis-top-image-sectioning-next': 'Plate Analysis',

#ANALYSIS STAGE CONVERT
'convert-progress': "Converting XML, this may take some time...",
'convert-dialog-title': "Select the XML slimmed file to work with",
'convert-xml-conversions': "Running conversions:",
'convert-xml-select-button': "Select...",
'convert-xml-select-label': "Choose Xml to be converted (output in same folder)",
'convert-xml-conversions-done': "Completed Conversions",
'convert-completed': "{0}: {1}",
'convert-completed-status': ["FAILED", "SUCCESS"],

#ANALYSIS STAGE EXTRACT
'extract-dialog': 'Select Directory...',
'extract-dialog-title': 'Select folder with analysis files',
'extract-tag-label': 'Understandable Job Tag',
'extract-launch-error': """Server refused or was not reachable""",
'extract-bad-directory': """The directory selected is not an analysis directory""",

#ANALYSIS STAGE IMAGE NORM MANUAL
'analysis-stage-image-sectioning-title':
"<span weight='heavy' size='large'>Marking out Plates</span>",

'analysis-stage-image-sectioning-help_text':
"""<i>If you have the same stuff in several images, mark out the plates
in the same order and watch the magic!</i>""",

#ANALYSIS TOP PLATE
'analysis-top-image-plate-next_plate': 'Next Plate',
'analysis-top-image-plate-next_image': 'Next Image',
'analysis-top-image-plate-next_done': 'Done!',
'analysis-top-image-plate-prev_plate': 'Previous Plate',
'analysis-top-image-plate-prev_norm': 'Previous Normalisation',

#ANALYSIS STAGE IMAGE PLATE
'analysis-stage-image-plate-title':
"<span weight='heavy' size='large'>Analyis of Plate {0}</span>",

'analysis-stage-image-plate-name': 'Plate Description/Media:',
'analysis-stage-image-plate-lock_selection': 'Lock Selection Size',
'analysis-stage-image-plate-selection-width': 'Selection Width:',
'analysis-stage-image-plate-selection-height': 'Selection Height:',
'analysis-stage-image-plate-colony-name': 'Name specific experiment:',
'analysis-stage-image-plate-log-button': 'Record data for this experiment',
'analysis-stage-image-plate-calibration': 'Independent measure',
'analysis-stage-image-plate-overshoot-warning': '\nThe colony is to thick.\nIt can\'t be reliably measured.\n\nIt should not be used!\n',

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
'analysis-top-auto-norm-and-section-prev-image': 'Previous Image',
'analysis-top-auto-norm-and-section-sel-images': 'Select Images',

#ANALYSIS STAGE AUTO NORMALISATION AND SECTIONING
'analysis-stage-auto-norm-and-section-title':
"<span weight='heavy' size='large'>Automatic Normalisation and Sectioning</span>",

'analysis-stage-auto-norm-and-section-file': 'Use already detected',
'analysis-stage-auto-norm-and-section-fixture': 'Detect:',
'analysis-stage-auto-norm-and-section-run': 'Run it!',
'analysis-stage-auto-norm-and-section-gs-title': 'Grayscale',
'analysis-stage-auto-norm-and-section-gs-help':
'Note that the curve should be monotonic (no bumps)',


}

specific_project = {
    'analysis-project-log_file': '',
    'analysis-project-log_file_dir': '',
    'analysis-project-output-default': 'analysis',
    'analysis-project-output-path': 'analysis',
    'analysis-project-plates': 0,
    'analysis-project-pinnings': (),
    'analysis-project-pinnings-from-file': (),
    'analysis-project-pinnings-active': 'file'
}

specific_first = {
    'output-directory': PATHS.experiment_root,
    'experiments-root': '',
    'experiment-prefix': '',
    'output-file': '',
    'use-local-fixture': False,
    'image-list-model': None,
    'meta-data': None,
    'run-cur-image': 0,
    'run-tot-images': 0,
    'run-position': 0,
    'run-complete': False,
    'run-error': None,
}

specific_transparency = {
    'mode': 'transparency',
    'fixture': True,
    'fixture-name': None,
    'stage': None,
    'manual-calibration-positions': None,
    'manual-calibration-values': None,
    'manual-calibration-target': None,
    'manual-calibration-grayscaleName': None,
    'manual-calibration-grayscale': False,
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
    'log-only-calibration': False,
    'log-interests': [None, None],
    'calibration-interests': ('Independent Count', 'Keys', 'Key Count'),
    'log-meta-features': ('image', 'plate-coords', ' plate-index',
                          'plate-media', 'selected-plate-area', 'strain'),
    'log-compartments-default': ('blob', 'background', 'cell'),
    'log-measures-default': ('pixelsum', 'area', 'mean', 'median', 'IQR',
                             'IQR_mean', 'centroid', 'perimeter'),
    'man-detection': [None, None],
}

specific_extract = {
    'path': '',
    'tag': ''
}

specific_log_book = {
    'manual-calibration': None,
    'log': list(),
    'log-interests': None,
    'images': list(),
    'plate-names': list(),
    'current-strain': None,
    'measures': list(),
    'calibration-measures': False,
    'calibration-measure-labels': [
        'independent-measure',
        'pixel-values',
        'pixel-counts'],
    'indie-count': None
}

specific_inspect = {
    'run-file': None,
    'analysis-dir': None,
    'experiment-dir': None,
    'uuid': None,
    'fixture': None,
    'prefix': None,
    'grid-images': [],
    'plate-names': [],
    'pinnings': None,
    'pinning-formats': None,
    'gridding-history': None,
    'gridding-in-history': None,
    'filezilla': False,
}
