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
'free-scanners': 'Free Scanners',
'running-experiments': 'Running Experiments',
'running-analysis': 'Running Analysis',
'collected-messages': 'Warnings & Errors',

#RUNNING EXPERIMENTS
'running-experiments-intro': 
"""<span size="x-large">Running experiments</span>

<span><i>Here you can see how the experiments progress and manually
terminate experiments before their planned end.</i></span>
""",
'running-experiments-stop': 'Terminate...',
}


composite_specific_model = {

'free-scanners': 0,
'scanner-procs': list(),
'analysis-procs': list(),
'messages': list(),
}
