#!/usr/bin/env python
"""The GTK-GUI model for the general layout"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# INTERNAL DEPENDENCIES
#

import generic.model_generic as model_generic

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
'window-title': 'Scan-o-matic v{0}'.format(__version__),

#PANEL

##ACTIONS
'panel-actions-title': 'Actions',
'panel-actions-analysis': 'Analysis',
'panel-actions-experiment': 'Experiment',
'panel-actions-calibration': 'Calibration',
'panel-actions-config': 'Application Config',
'panel-actions-qc': 'Quality Control',
'panel-actions-quit': 'Quit',

#MAIN

##PAGE TITLES
'content-page-title-analysis': 'Analysis',
'content-page-title-experiment': 'Experiment',
'content-page-title-calibration': 'Calibration',
'content-page-title-config': 'App Config',
'content-page-title-qc': 'Quality Control',

#DIALOGS
'content-page-close': ("Closing content page will cause all unsaved progress"
    " to be lost.\n\nDo you wish to continue?"),
'content-app-close-orphan-warning': ("You are about to make some subprocesses"
    " orphans.\nThat's a no-no and could well mean that the next person,\n"
    "oblivious to your cruelty happens to terminate them for good.\n"
    "Destroying someones work in the process.\n\n"
    "Are you really really super sure you want to quit?"),
}
