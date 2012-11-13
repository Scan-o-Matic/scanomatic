#
# DEPENDENCIES
#

import numpy as np

#
# INTERNAL DEPENDENCIES
#

import src.model_generic as model_generic

#
# FUNCTIONS
#

def get_gui_model():

    m = Composite_Subproc_GUI_Model(private_model=composite_model,
        generic_model=model_generic.get_model())

    return m

def get_composite_specific_model():

    m = Subproc_Generic_Specific_Model(
        private_model=copy_model(composite_specific_model))

    m.add_counter('running-scanners', 'scanner-procs')
    m.add_counter('running-analysis', 'analysis-procs')
    m.add_counter('collected-messages', 'messages')

    return m

def copy_model(model):

    return model_generic.copy.deepcopy(model)

#
# SUBCLASSING
#

class Composite_Subproc_GUI_Model(model_generic.Model): pass
class Subproc_Generic_Specific_Model(model_generic.Model): pass

#
# MODELS
#

composite_model = {

'composite-stat-title': 'Status',
'composite-stat-type-header': '<u>Process</u>',
'composite-stat-count-header': '<u>Running</u>',
'running-scanners': 'Scanners',
'running-analysis': 'Analysis',
'collected-messages': 'Warnings & Errors',

}


composite_specific_model = {

'scanner-procs': list(),
'analysis-procs': list(),
'messages': list(),
}
