#
# INTERNAL DEPENDENCIES
#

import src.model_generic as model_generic

#
# FUNCTIONS
#

def get_gui_model():

    m = Config_GUI_Model(private_model=model,
        generic_model=model_generic.get_model())

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

##DESKTOP SHORT-CUT
'config-desktop-short_cut': 'If you want a desktop shortcut...',
'config-desktop-short_cut-make':'click me!',
'config-desktop-short_cut-made': 'Success!\nShort-cut was made..',

##BACKUP LOGS AND STATE
'config-log-save': 
"""If you run into problems, this allows for backup of logs and program state for later debugging""",
'config-log-save-button': 'Make back-up...',
'config-log-save-dialog': 'Select place and friendly name for logs',
'config-log-file-filter': {'filter_name': '.tar.gz - files',
'mime_and_patterns': (('application/tar', '*.tar.gz'),)},
'config-log-save-done': 'Success! Save complete!',

}
