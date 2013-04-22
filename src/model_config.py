#
# INTERNAL DEPENDENCIES
#

import src.model_generic as model_generic

#
# FUNCTIONS
#

def get_gui_model(paths=None):

    m = Config_GUI_Model(private_model=model,
        generic_model=model_generic.get_model(), paths=paths)

    return m


def copy_model(model):

    return model_generic.copy.deepcopy(model)

#
# SUBCLASSING
#

class Config_GUI_Model(model_generic.Model): pass

#
# MODELS
#

model = {

#TOP
'config-title': '<span size="x-large"><u>Application Config</u></span>',

#STAGE

##INSTALL
'config-install': 'Installation stuff',

###DESKTOP SHORT-CUT
'config-desktop-short_cut': 'If you want a desktop shortcut...',
'config-desktop-short_cut-make':'click me!',
'config-desktop-short_cut-made': 'Success!\nShort-cut was made..',

###UPDATE
'config-update': 'Update',
'config-update-button': 'Check for Updates',
'config-update-restart': 'Restart for updates to take place',
'config-update-no-sh': 'Please install python module sh via easy_install',
'config-update-up_to_date': 'Your system was allready up to date',
'config-update-warning': 'Someone has been tinkering with your copy, get a hacker to fix that so you can download the update',
'config-update-success': 'Update was successful, but you need to restart for the changes to take place',

##BACKUP AND ERRORS
'config-backup': 'Backup and Errors',

###BACKUP LOGS AND STATE
'config-log-save':
"""If you run into problems, this allows for backup of logs and program state for later debugging""",
'config-log-save-button': 'Make back-up...',
'config-log-save-dialog': 'Select place and friendly name for logs',
'config-log-file-filter': {'filter_name': '.tar.gz - files',
'mime_and_patterns': (('application/tar', '*.tar.gz'),)},
'config-log-save-done': 'Success! Save complete!',

##REMOVE FIXUTRES
'config-fixtures': 'Remove fixture(s)',
'config-fixture-remove': 'Remove',
'config-fixture-dialog': 'Are you sure you wish to remove {0}?',

##SETTINGS
'config-settings': 'Settings',

###POWER MANAGER
'config-pm': 'Power Manager Type',
'config-pm-no': 'NO PM',
'config-pm-usb': 'USB',
'config-pm-lan': 'LAN/NETWORK',

##SCANNERS
'config-scanners': 'Number of Scanners',

##EXPERIMENTS-ROOT
'config-settings-experiments-root': 'Select Experiments Root',

#SAVE
'config-settings-save': 'Save and Apply Settings',
}
