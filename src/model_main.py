#
# INTERNAL DEPENDENCIES
#

import src.model_generic as model_generic

#
# FUNCTIONS
#

def load_app_model():
    return model_generic.Model(private_model=model)


def copy_model(model):

    return model_generic.copy.deepcopy(model)
#
# MODELS
#

model = {
'window-title': 'Scan-o-matic',

#PANEL

##ACTIONS
'panel-actions-title': 'Actions',
'panel-actions-analysis': 'Analysis',
'panel-actions-experiment': 'Experiment',
'panel-actions-calibration': 'Calibration',
'panel-actions-quit': 'Quit',

#MAIN

##PAGE TITLES
'content-page-title-analysis': 'Analysis',
'content-page-title-experiment': 'Experiment',
'content-page-title-calibration': 'Calibration',

#DIALOGS
'content-page-close': 'Closing content page will cause all unsaved progress to be lost.\n\nDo you wish to continue?'
}
