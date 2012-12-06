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
'config-desktop-short_cut': 'If you want a desktop shortcut...',
'config-desktop-short_cut-make':'click me!',
'config-desktop-short_cut-made': 'Success!\nShort-cut was made..',
}
