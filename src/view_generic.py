#!/usr/bin/env python
"""The GTK-GUI view"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
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
# STATIC GLOBALS
#

PADDING_LARGE = 10
PADDING_MEDIUM = 4
PADDING_SMALL = 2

#
# FUNCTIONS
#

def select_dir(title):

    d = gtk.FileChooserDialog(title=title, 
        action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER, 
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
        gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    res = d.run()
    file_list = d.get_filename()
    d.destroy()

    if res == gtk.RESPONSE_APPLY:

        return file_list

    else:

        return None


def select_file(title, multiple_files=False, file_filter=None):

    d = gtk.FileChooserDialog(title=title, 
        action=gtk.FILE_CHOOSER_ACTION_OPEN, 
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
        gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    d.set_select_multiple(multiple_files)

    if file_filter is not None:

        f = gtk.FileFilter()
        f.set_name(file_filter['filter_name'])
        for m, p in file_filter['mime_and_patterns']:
            f.add_mime_type(m)
            f.add_pattern(p)
        d.add_filter(f)

    res = d.run()
    file_list = d.get_filenames()
    d.destroy()

    if res == gtk.RESPONSE_APPLY:

        return file_list

    else:

        return list()

def save_file(title, multiple_files=False, file_filter=None):

    d = gtk.FileChooserDialog(title=title, 
        action=gtk.FILE_CHOOSER_ACTION_SAVE, 
        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
        gtk.STOCK_APPLY, gtk.RESPONSE_APPLY))

    d.set_select_multiple(multiple_files)

    if file_filter is not None:

        f = gtk.FileFilter()
        f.set_name(file_filter['filter_name'])
        for m, p in file_filter['mime_and_patterns']:
            f.add_mime_type(m)
            f.add_pattern(p)
        d.add_filter(f)

    res = d.run()
    file_list = d.get_filenames()
    d.destroy()

    if res == gtk.RESPONSE_APPLY:

        return file_list

    else:

        return list()


def overwrite(text, file_name, window):

    dialog = gtk.MessageDialog(window,
                    gtk.DIALOG_DESTROY_WITH_PARENT,
                    gtk.MESSAGE_WARNING, gtk.BUTTONS_NONE,
                    text.format(file_name))

    dialog.add_button(gtk.STOCK_NO, False)
    dialog.add_button(gtk.STOCK_YES, True)

    dialog.show_all()

    resp = dialog.run()

    dialog.destroy()

    return resp

def dialog(window, text, d_type="info", yn_buttons=False):

    d_types = {'info': gtk.MESSAGE_INFO, 'error': gtk.MESSAGE_ERROR,
        'warning': gtk.MESSAGE_WARNING, 'question': gtk.MESSAGE_QUESTION,
        'other': gtk.MESSAGE_OTHER}

    if d_type in d_types.keys():

        d_type = d_types[d_type]

    else:

        d_type = gtk.MESSAGE_INFO

    d = gtk.MessageDialog(window, gtk.DIALOG_DESTROY_WITH_PARENT,
        gtk.MESSAGE_INFO, gtk.BUTTONS_NONE,
        text)

    if yn_buttons:

        d.add_button(gtk.STOCK_YES, True)
        d.add_button(gtk.STOCK_NO, False)

    else:

        d.add_button(gtk.STOCK_OK, -1)

    result = d.run()

    d.destroy()

    return result

#
# CLASSES
#


class Pinning(gtk.VBox):

    def __init__(self, controller, model, project_veiw, 
            plate_number, specific_model, pinning = None):

            self._model = model
            self._specific_model = specific_model
            self._controller = controller
            self._project_view = project_veiw

            super(Pinning, self).__init__()

            label = gtk.Label(
                model['analysis-stage-project-plate-label'].format(
                plate_number))


            self.pack_start(label, False, False, PADDING_SMALL)

            if pinning is None:

                pinning = specific_model['analysis-project-pinning-default']

            self.dropbox = gtk.combo_box_new_text()                   

            def_key = 0
            ref_matrices = model['pinning-matrices-reversed']
            for i, m in enumerate(sorted(model['pinning-matrices'].keys())):

                self.dropbox.append_text(m)

                if pinning in ref_matrices and ref_matrices[pinning] == m:
                    def_key = i

            self.dropbox.set_active(def_key)
            
            self.dropbox_signal = self.dropbox.connect("changed",
                self._controller.project.set_pinning,
                plate_number)


            self.pack_start(self.dropbox, True, True, PADDING_SMALL)

    def set_sensitive(self, val):

        if val is None:
            val = False

        self.dropbox.set_sensitive(val)


    def set_pinning(self, pinning):

        orig_key = self.dropbox.get_active()
        new_key = -2
        pinning_text = self._model['pinning-matrices-reversed'][pinning]

        for i, m in enumerate(sorted(self._model['pinning-matrices'].keys())):

            if pinning_text == m:
                new_key = i

        if new_key != orig_key:

            self.dropbox.handler_block(self.dropbox_signal)
            self.dropbox.set_active(new_key)
            self.dropbox.handler_unblock(self.dropbox_signal)
