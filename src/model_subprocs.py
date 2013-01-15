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
'running-experiments-stop-warning':
"""<span size="x-large">Are you sure you want to stop experiment '{0}'?</span>

<span><i>Stopping is irreversible and will act as soon as current scan is aquired.
To stop, write 'stop' below and then presss the yes-button.</i></span>
""",
'running-experiments-stopping': 'Stopping...',

#RUNNING ANALYSIS
'running-analysis-intro':
"""<span size="x-large">Running analysis</span>

<span><i>Here you can see which analysis that are running,
and when to expect the results.</i></span>
""",
'running-analysis-running': 'Running analysis start-up procedures',
'running-analysis-current': 'Analysing {0} of {1}',
'running-analysis-done': 'Done!',
'running-analysis-progress-bar-elapsed': 'Elapsed time: {0:.1f} min',
'running-analysis-progress-bar-eta': "Expected to finnish in {0:.2f} h",
 
}


composite_specific_model = {

'free-scanners': 0,
'scanner-procs': list(),
'analysis-procs': list(),
'messages': list(),
}
