"""The Main Controller"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.new_model_generic as new_model_generic


class Model(new_model_generic.Model):
    #
    # MODEL DEFAULTS
    #

    _PRESETS_APP = {
        'app-title': "Scan-o-Matic: Quality Control",
        'quit-unsaved': "You have unsaved work, are you sure you want to quit?",

    }

    _PRESETS_STAGE = {

        'debug-mode': False,

        'msg-multiSelecting': "Currently using multi-seletion",
        'showRaw': True,
        'phenotype': None,
        'phenotyper': None,
        'phenotyper-path': "",
        '_plate': 0,
        'fixedColors': (None, None),
        'colorsAll': True,
        '_selectionFilter': None,

        'plates': None,
        '_platesHaveUnsaved': None,
        'subplateSelected': np.zeros((2, 2), dtype=bool),
        'selection_patches': None,  # dict(),
        'selectOnAllPhenotypes': True,

        'button-load-data': "Load experiment data",
        'button-load-meta': "Load meta-data",

        'load-fail-text': """The project could not be loaded, either files are
    corrupt or missing""",
        'phenotype-fail-text': """The phenotype has not been extracted, maybe
    re-run extraction?""",
        'loaded-text': "Loaded Project",
        'no-plates-loaded': "-Not Loaded-",

        'label-plate': "Plate:",
        'plate-save': "Save Plate Image",

        'colors': 'Set Colors From:',
        'color-one-plate': "This plate",
        'color-all-plates': "All plates",
        'color-fixed': "Values",
        'color-fixed-update': "Update",

        'curve-save': "Save Curve Image",

        'selections-section': "Selections & Undo",
        'unselect': "Unselect All",

        'badness-label': 'Quality Index',
        'removeCurvesPhenotype': "Delete marked, this phenotype",
        'removeCurvesAllPhenotypes': "Delete marked, all phenotypes",
        'undo': "Undo",
        'unremove-text': """Unfortunately the undo history is empty.
    Do you wish to restore all removed curves for all phenotypes for
    the current plate?""",

        'auto-selecting': True,

        'multi-select-phenotype': "Select based on phenotype",
        'multi-sel-lower': "Lower:",
        'multi-sel-higher': "Upper:",

        'subplate-selection': "Subplate Select",
        'subplate-0-0': "Upper Left",
        'subplate-0-1': "Upper Right",
        'subplate-1-0': "Lower Left",
        'subplate-1-1': "Lower Right",

        'frame-normalize': "References / Normalize",
        'references': "Selected plate(s) as Reference",
        'normalize': "Normalize",

        'save-absolute': "Save the absolute phenotypes",
        'save-relative': "Save the normalized phenotypes",

        'saved-absolute': "Absolute Phenotypes Saved at '{0}'",
        'saved-normed': "Normalized Relative Phenotypes Saved at '{0}'",

        'save-state': "Save State",
        'reference-positions': None,
        'set-references': "Reference Plate Set",

        'no-reference': "No Reference",
        'yes-reference': "Ref: ",
        'reference-offset-names': ("Up Left", "Up Right", "Low Left", "Low Right"),

        'image-saved': "Image saved at '{0}'",

        'saveTo': "Save to...",

        'meta-data': None,
        'meta-data-files': "Select Meta-Data Files",
        'meta-data-loaded': "Meta-Data Loaded!",
        'meta-data-info-column': 0,

        'hover-position': 'Row {0}, Column {1}',
        'load-data-dir': "Select Directory With Data Files",

        'normalized-data': None,
        'normalized-index-offset': None,
        'normalized-phenotype-names': dict(),
        'normalized-phenotype': "2D Normalized {0}",
        'normalized-text': "2D Normalization Done",
        'norm-alg-in-log-text': "Algorithm in Log",

        'norm-alg-in-log': False,
        'norm-outlier-fillSize': None,
        'norm-outlier-k': 2.0,
        'norm-outlier-p': 10,
        'norm-outlier-iterations': 10,
        'norm-smoothing': 1.0,
        'norm-spline-seq': ('cubic', 'nearest'),
        'norm-use-initial-values': False,
        'norm-use-initial-text': "Use initial values as guide",

        'norm-ref-usage-threshold': 0.95,
        'norm-ref-CV-threshold': 0.236,  # Magic number estimated by Dr. A. Skyman
        'ref-warning-head': """Some plates may have bad quality and should be
    discarded, or you failed to remove all bad positions:\n""",
        'ref-bad-plate': """Plate {0} has only used {1:.1f} percent of references
    and has a CV of {2:.3f}""",
    }

    def visibleValues(self):

        if (self['phenotype'] < self['phenotyper'].nPhenotypeTypes):

            V = self['phenotyper'].phenotypes[self['plate']][
                ..., self['phenotype']].ravel()

        else:

            V = self['normalized-data'][self['plate']][
                ..., self['absPhenotype']].ravel()

        return V[np.isfinite(V)]

    def visibleMin(self):

        v = self['visibleValues']
        if (v.size):
            return v.min()
        else:
            return 0

    def visibleMax(self):

        v = self['visibleValues']
        if (v.size):
            return v.max()
        else:
            return 0

    def absPhenotype(self):

        if (self['phenotype'] < self['phenotyper'].nPhenotypeTypes or
                self['normalized-index-offset'] is None):

            return self['phenotype']

        else:

            return self['phenotype'] - self['normalized-index-offset']

    def numberOfSelections(self):

        return self['plate_selections'][..., self['absPhenotype']].sum()

    def selectionCoordinates(self):

        return zip(*np.where(self['plate_selections'][...,
                                                      self['absPhenotype']]))

    def selectionWhere(self):

        return np.where(self['plate_selections'][..., self['absPhenotype']])

    def multiSelecting(self):

        return self['numberOfSelections'] > 1

    def showSmooth(self):

        return not self['multiSelecting']

    def showGTregLine(self):

        return not self['multiSelecting']

    def showModelLine(self):

        return not self['multiSelecting']

    def showFitValue(self):

        return not self['multiSelecting']

    def showGT(self):

        return not self['multiSelecting']

    def plate_exists(self):

        return (self['phenotyper'] is not None and
                self['plate'] is not None and
                self['phenotyper'][self['plate']] is not None)

    def plate_shapes(self):

        if self['phenotyper'] is None:
            return None

        return [p is None and None or p.shape[:2] for p in self['phenotyper']]

    def plate_size(self):

        if (self['phenotyper'] is None or
                self['phenotyper'][self['plate']] is None):

            return 0

        pShape = self['phenotyper'][self['plate']].shape

        return pShape[0] * pShape[1]

    def plate_selections(self):

        if self['phenotyper'] is None:
            return None

        sf = self['_selectionFilter']
        if sf is None or not all(f.shape[:2] == s.shape[:2] for f, s in zip(
                sf, self['phenotyper'])):

            sf = [np.zeros(s.shape[:2] + (self['phenotyper'].nPhenotypeTypes,),
                           dtype=bool) for s in self['phenotyper']]

            self['_selectionFilter'] = sf

        return sf[self['plate']][..., self['selectOnAllPhenotypes']
                                 and slice(None) or self['absPhenotype']]

    def removed_filter(self):

        return (self['phenotyper'] is None and None or
                self['phenotyper'].getRemoveFilter(self['plate']))

    def plate_has_removed(self):

        if self['phenotyper'] is None:
            return False
        else:
            return self['phenotyper'].hasRemoved(self['plate'])

    def platesHaveUnsaved(self):

        self._initUnsaved()
        return self['_platesHaveUnsaved']

    def unsaved(self):

        self._initUnsaved()
        return any(self['_platesHaveUnsaved'])

    def _initUnsaved(self):

        if (self['_platesHaveUnsaved'] is None or
                self['_platesHaveUnsaved'].size == 0):

            if self['phenotyper'] is None:
                self['_platesHaveUnsaved'] = np.array([])
            else:
                self['_platesHaveUnsaved'] = np.array(
                    [False for _ in self['phenotyper']])

    def badSortingPhenotypes(self):

        p = self['phenotyper']
        if p is None:
            return tuple()

        return (p.PHEN_GT_ERR, p.PHEN_GT_POS,
                p.PHEN_GT_2ND_ERR, p.PHEN_GT_2ND_POS,
                p.PHEN_FIT_VALUE)

    def plate(self):

        if (self['phenotyper'] is None):
            raise TypeError("Phenotyper cannot be None")
        if (self['_plate'] is None):
            raise TypeError("Plate cannot be None")

        return self['_plate']
