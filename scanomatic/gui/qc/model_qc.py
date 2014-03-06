import numpy as np


class NewModel(object):

    def __init__(self, presets=dict()):

        self._values = presets

    def __getitem__(self, key):

        if key in self._values:
            return self._values[key]

        elif hasattr(self, key):

            return getattr(self, key)()

        else:

            raise KeyError

    def __setitem__(self, key, value):

        self._values[key] = value

    @classmethod
    def LoadAppModel(cls):

        return cls(presets=dict(_appPresets.items() + _stagePresets.items()))

    @classmethod
    def LoadStageModel(cls):

        return cls(presets=_stagePresets)

    def multiSelecting(self):

        return len(self['curSelection']) > 0

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

    def plate_selections(self):

        if self['phenotyper'] is None:
            return None

        sf = self['_selectionFilter']
        if sf is None or not all(f.shape[:2] == s.shape for f, s in zip(
                sf, self['phenotyper'][self['plate']])):

            sf = [np.zeros(s.shape[:2] + (self['phenotyper'].nPhenotypeTypes,),
                           dtype=bool) for s in self['phenotyper']]

            self['_selectionFilter'] = sf

        return sf[self['plate']][..., self['selectOnAllPhenotypes']
                                 and self['phenotype'] or slice(None)]

    def removed_filter(self):

        rf = self['_removeFilter']
        if rf is None or not all(f.shape[:2] == s.shape for f, s in zip(
                rf, self['phenotyper'][self['plate']])):

            rf = [np.zeros(s.shape[:2] + (self['phenotyper'].nPhenotypeTypes,),
                           dtype=bool) for s in self['phenotyper']]

            self['_removeFilter'] = rf

        return rf[self['plate']]

    def removed_filter_phenotype(self):

        return self['removed_filter'][..., self['phenotype']]

    def unsaved(self):

        rf = self['_removeFilter']
        sf = self['_selectionFilter']
        return (rf is not None and any([f.any() for f in rf]) or
                sf is not None and any([f.any() for f in sf]))

#
# MODEL DEFAULTS
#

_appPresets = {
    'app-title': "Scan-o-Matic: Quality Control",
    'quit-unsaved': "You have unsaved work, are you sure you want to quit?",

}

_stagePresets = {
    'msg-multiSelecting': "Currently using multi-seletion",
    'showRaw': True,
    'phenotype': None,
    'phenotyper': None,
    'plate': None,
    'fixedColors': (None, None),
    'colorsAll': True,
    '_selectionFilter': None,
    '_removeFilter': None,
    'plates': None,
    'subplateSelected': np.zeros((2, 2), dtype=bool),
    'selection_patches': dict(),
    'selectOnAllPhenotypes': True,

    'button-load-data': "Load experiment data",
    'button-load-meta': "Load meta-data",

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
    'removeCurvesPhenotype': "Delete marked, this phenotype",
    'removeCurvesAllPhenotypes': "Delete marked, all phenotypes",
    'undo': "Undo",

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

    'reference-positions': None,
    'set-references': "Reference Plate Set",

    'image-saved': "Image saved at '{0}'",

    'saveTo': "Save to...",

    'meta-data-files': "Select Meta-Data Files",
    'meta-data-loaded': "Meta-Data Loaded!",

    'load-data-dir': "Select Directory With Data Files",
}
