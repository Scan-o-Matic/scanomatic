"""The GTK-GUI view"""
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

import pygtk
pygtk.require('2.0')
import gtk

#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.view_generic as view_generic

#
# STATIC GLOBALS
#

from scanomatic.gui.generic.view_generic import \
    PADDING_SMALL

#
# CLASSES
#


class Analysis_Stage_About(gtk.Label):

    def __init__(self, controller, model):

        self._controller = controller
        self._model = model

        super(Analysis_Stage_About, self).__init__()

        self.set_justify(gtk.JUSTIFY_LEFT)
        self.set_markup(model['analysis-stage-about-text'])

        self.show()


class Analysis_Top_Root(view_generic.Top):

    def __init__(self, controller, model):

        super(Analysis_Top_Root, self).__init__(controller, model)

        button = gtk.Button()
        button.set_label(model["analysis-top-root-project_button-text"])
        button.set_sensitive(False)
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "project")

        """
        button = gtk.Button()
        button.set_label(model["analysis-top-root-color_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "colour")
        """

        button = gtk.Button()
        button.set_label(model["analysis-top-root-1st_pass-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "1st_pass")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-inspect-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "inspect")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-convert"])
        self.pack_start(button, expand=False, fill=False,
                        padding=PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "convert")

        button = gtk.Button()
        button.set_label(model["analysis-top-root-features"])
        self.pack_start(button, expand=False, fill=False,
                        padding=PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "extract")

        self.pack_start(gtk.VSeparator(), expand=False, fill=False,
                        padding=PADDING_SMALL)

        button = gtk.Button()
        button.set_label(model["analysis-top-root-tpu_button-text"])
        self.pack_start(button, False, False, PADDING_SMALL)
        button.connect("clicked", controller.set_analysis_stage, "transparency")

        self.show_all()
