#!/usr/bin/env python
"""The Analysis Controller"""
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

import os
import types
import gobject
import threading
import numpy as np
import copy
from subprocess import Popen

#
# INTERNAL DEPENDENCIES
#

import src.model_analysis as model_analysis
import src.view_analysis as view_analysis
import src.controller_generic as controller_generic
import src.resource_os as resource_os
import src.resource_project_log as resource_project_log
import src.analysis_wrapper as a_wrapper
import src.resource_fixture_image as resource_fixture_image
import src.resource_image as resource_image

#
# EXCEPTIONS
#

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class Unknown_Log_Request(Exception): pass
class UnDocumented_Error(Exception): pass

#
# CLASSES
#

class Analysis_Controller(controller_generic.Controller):

    def __init__(self, main_controller, logger=None):


        super(Analysis_Controller, self).__init__(
                    main_controller, logger=logger)

        self.transparency = Analysis_Transparency_Controller(
                self, view=self._view, model=self._model,
                logger=logger)

        self._specific_controller = None

        self.fixture = None

    def _get_default_view(self):

        return view_analysis.Analysis(self, self._model)

    def _get_default_model(self):

        return model_analysis.get_gui_model()

    def _callback(self, user_data):

        if user_data is not None:

            if user_data['view-data'] is None and 'cb' not in user_data:

                user_data['cb'] = 9

            if user_data['thread'].is_alive() == False:

                user_data['view-function'](user_data['view-complete'])
                user_data['view'].run_release()

                if 'complete-function' in user_data:

                    user_data['complete-function']()

                return False

            else:

                if user_data['view-data'] is None:

                    user_data['cb'] += 1
                    user_data['view-function'](1 - np.exp(-0.01 * user_data['cb']))

                else:

                    user_data['view-function'](user_data['view-data'])

        if user_data is None:

            print "LOST!"
            return False

        gobject.timeout_add(250, self._callback, user_data)
        return None

    def get_available_fixtures(self):

        return self.get_top_controller().fixtures.get_names()

    def set_analysis_stage(self, widget, *args, **kwargs):

        if len(args) < 1:

            raise Bad_Stage_Call()

        else:

            stage_call = args[0]

            view = self.get_view()
            model = self.get_model()

            if view is None:

                raise No_View_Loaded()

            if stage_call == "about":

                view.set_top()
                view.set_stage()

            elif stage_call == "project":

                self._specific_controller = \
                        Analysis_Project_Controller(self,
                        view=self._view, model=self._model,
                        logger=self._logger)

                self.add_subcontroller(self._specific_controller)

                view.set_top(view_analysis.Analysis_Top_Project(
                    self._specific_controller, model))

                view.set_stage(view_analysis.Analysis_Stage_Project(
                    self._specific_controller, model))

            elif stage_call == "1st_pass":

                self._specific_controller = Analysis_First_Pass(
                    self, view=view, model=self._model,
                    logger=self._logger)

                self.add_subcontroller(self._specific_controller)

                view.set_top(
                    view_analysis.Analysis_First_Pass_Top(
                        self._specific_controller, model))

                view.set_stage(
                    view_analysis.Analysis_Stage_First_Pass(
                        self._specific_controller, model))

            elif stage_call == "transparency":

                #IF CALLED WITHOUT MODEL CREATE ONE
                if len(args) < 2:
                    self.transparency.build_blank_specific_model()

                self.transparency._specific_model['stage'] = 'image-selection'

                view.set_top(view_analysis.Analysis_Top_Image_Selection(
                                self, model,
                                self.transparency.get_specific_model(),
                                self.transparency))

                view.set_stage(view_analysis.Analysis_Stage_Image_Selection(
                                self, model,
                                self.transparency.get_specific_model(),
                                self.transparency))

            elif stage_call == "colour":

                raise Not_Yet_Implemented()
        
            elif stage_call == "normalisation":

                specific_model = args[1]
                self.set_unsaved()


                if specific_model['mode'] == 'transparency':

                    specific_model['image'] += 1

                    if specific_model['image'] >= len(
                                specific_model['images-list-model']):

                        raise Bad_Stage_Call("Image position overflow")

                    if specific_model['fixture']:

                        specific_model['stage'] = 'auto-calibration'
                        specific_model['plate'] = -1

                        model['fixtures'] = self.get_available_fixtures()

                        top = view_analysis.Analysis_Top_Auto_Norm_and_Section(
                            self, model,
                            specific_model,
                            self.transparency)

                        if specific_model['image'] > 0:
                            message = 'previous-image'
                            label = model[
                                'analysis-top-auto-norm-and-section-prev-image']
                        else:
                            message = 'select-images'
                            label = model[
                                'analysis-top-auto-norm-and-section-sel-images']

                        top.pack_back_button(label, 
                            self.transparency.step_back, message)

                        view.set_top(top)

                        view.set_stage(
                            view_analysis.Analysis_Stage_Auto_Norm_and_Section(
                            self, model,
                            specific_model,
                            self.transparency))

                    else:

                        specific_model['stage'] = 'manual-calibration'

                        view.set_top(
                            view_analysis.Analysis_Top_Image_Normalisation(
                            self, model,
                            specific_model,
                            self.transparency))

                        view.set_stage(
                            view_analysis.Analysis_Stage_Image_Norm_Manual(
                            self, model,
                            specific_model,
                            self.transparency))

                elif specific_model['mode'] == 'colour':

                    raise Not_Yet_Implemented((stage_call, specific_model['mode']))

                else:

                    raise Bad_Stage_Call(stage_call)

            elif stage_call == "sectioning":

                self.set_unsaved()
                specific_model = args[1]
                specific_model['stage'] = 'sectioning'
                specific_model['plate-coords'] = list()

                view.set_top(
                    view_analysis.Analysis_Top_Image_Sectioning(
                    self, model,
                    specific_model,
                    self.transparency))

                view.set_stage(
                    view_analysis.Analysis_Stage_Image_Sectioning(
                    self, model,
                    specific_model,
                    self.transparency,
                    self.get_window()))

                specific_model['plate'] = -1

            elif stage_call == "plate":

                self.set_unsaved()

                specific_model = args[1]
                specific_model['plate'] += 1
                specific_model['stage'] = 'plate'

                if self.transparency._log is None:
                    self.transparency._log = Analysis_Log_Controller(
                        self, self._model, specific_model)

                else:

                    self.transparency._log.set_view()
                    self.transparency._log._model['current-strain'] = None

                if specific_model['plate'] < len(specific_model['plate-coords']):

                    coords = specific_model['plate-coords'][
                            specific_model['plate']]

                    image = specific_model['image']
                    
                    if image in specific_model['auto-transpose']:


                        specific_model['plate-im-array'] = \
                            specific_model['auto-transpose'][image]\
                            .get_transposed_im(
                            specific_model['image-array'][
                            coords[0][1]: coords[1][1],
                            coords[0][0]: coords[1][0]])

                        specific_model['plate-is-normed'] = True

                    else:

                        specific_model['plate-im-array'] = \
                            specific_model['image-array'][
                            coords[0][1]: coords[1][1],
                            coords[0][0]: coords[1][0]]

                        specific_model['plate-is-normed'] = False

                    top = view_analysis.Analysis_Top_Image_Plate(
                        self, model,
                        specific_model,
                        self.transparency)

                    if specific_model['plate'] > 0:
                        message = 'previous-plate'
                        label = model[
                            'analysis-top-image-plate-prev_plate']
                    else:
                        message = 'normalisation'
                        label = model[
                            'analysis-top-image-plate-prev_norm']

                    top.pack_back_button(label, 
                        self.transparency.step_back, message)
                    view.set_top(top)
                    top.set_allow_next(True)

                    view.set_stage(
                        view_analysis.Analysis_Stage_Image_Plate(
                        self, model,
                        specific_model,
                        self.transparency))
                    
                    view.get_stage().run_lock_select_check()

            elif stage_call == "log_book":

                specific_model = args[1]
                specific_model['stage'] = 'done'

                view.set_top(
                    view_analysis.Analysis_Top_Done(
                    self, model))

                self.transparency._log.set_view()
                view.set_stage(
                    self.transparency._log.get_view())

            else:

                raise Bad_Stage_Call(stage_call)


class Analysis_First_Pass(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, logger=None):

        super(Analysis_First_Pass, self).__init__(
                parent, 
                view=view, model=model, logger=logger)

        self._specific_model = None

    def start(self, *args, **kwargs):

        print "Start", args, kwargs


class Analysis_Image_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, logger=None):

        super(Analysis_Image_Controller, self).__init__(
                parent, 
                view=view, model=model, logger=logger)

        self._specific_model = None
        self._log = None

    def step_back(self, widget, message):

        sm = self._specific_model

        if message == 'select-images':

            sm['image'] = -1
            sm['plate'] = -1
            stage_call = sm['mode']

        elif message == 'previous-image':

            sm['image'] -= 2
            sm['plate'] = -1

            stage_call = 'normalisation'

        elif message == 'previous-plate':

            sm['plate'] -= 2
            stage_call = 'plate'

        elif message == 'normalisation':

            sm['image'] -= 1
            sm['plate'] = -1
            stage_call = 'normalisation'

        self._parent.set_analysis_stage(None, stage_call, sm)

    def execute_fixture(self, widget, data):

        view, specific_model = data
        view.run_lock()
        fixture_path = self.get_top_controller().paths.fixtures

        self.fixture = resource_fixture_image.Fixture_Image(
            specific_model["fixture-name"],
            image_path=specific_model['images-list-model'][
            specific_model['image']][0],
            fixture_directory=fixture_path
            )

        thread = threading.Thread(target=self.fixture.threaded)

        thread.start()

        gobject.timeout_add(250, self._parent._callback, {
            'view': view,
            'view-function': view.set_progress,
            'view-data':None,
            'view-complete': 1.0,
            'complete-function': self.set_grayscale,
            'thread': thread})

    def _get_scale_slice(self, the_slice, flip_coords=False, factor=4):

        data_sorted = zip(*map(sorted, zip(*the_slice)))
        ret = [[factor*p[flip_coords], factor*p[not(flip_coords)]]
                for p in data_sorted]

        return ret

    def get_previously_detected(self, view, specific_model):

        image=specific_model['images-list-model'][
                specific_model['image']][0]

        im_dir = os.sep.join(image.split(os.sep)[:-1])

        image = image.split(os.sep)[-1]

        extension = ".log"

        log_files = [im_dir + os.sep + f for f in os.listdir(im_dir) 
                        if f.lower().endswith(extension)]

        data = None

        
        for f in log_files:

            data = resource_project_log.get_image_from_log_file(f, image)

            if data is not None:

                break

        if data is None:

            view.set_detect_lock(False)

        else:

            i = 0

            plate_coords = dict()

            #Backwards compatible with spell error in early log-files
            gs_i_key = [k for k in data if 'grayscale_in' in k][0]

            while 'plate_{0}_area'.format(i) in data:

                plate_coords[i] = self._get_scale_slice(
                    data['plate_{0}_area'.format(i)],
                    flip_coords=True)
                i += 1

            specific_model['plate-coords'] = plate_coords

            self.set_auto_grayscale(data['grayscale_values'],
                data[gs_i_key])

    def set_no_auto_norm(self):

        sm = self._specific_model
        del sm['auto-transpose'][sm['image']]
        self.get_view().get_top().set_allow_next(False)

    def set_fixture(self, view, fixture_name, specific_model):

        specific_model['fixture-name'] = fixture_name

    def set_auto_grayscale(self, vals, indices):

        sm = self._specific_model

        sm['auto-transpose'][sm['image']] = \
            resource_image.Image_Transpose(
            gs_values=vals,
            gs_indices=indices)
            
        self.get_view().get_top().set_allow_next(True)

    def set_grayscale(self):

        if self.fixture is not None:

            gs_targets, gs = self.fixture['grayscale']
            self.set_auto_grayscale(gs, gs_targets) 
            self.get_view().get_stage().set_image()

            pl = self.fixture.get_plates()
            version = self.fixture['fixture']['version']
            if version is None or version < 0.997:
                back_compatible=True
            else:
                back_compatible=False

            plate_coords = dict()
            if pl is not None:

                s_pattern = "plate_{0}_area"
 
                for i, p in enumerate(pl):

                    plate_coords[i] = p

            self._specific_model['plate-coords'] = plate_coords

        else:

            raise UnDocumented_Error()

    def set_images_has_fixture(self, widget, *args, **kwargs):

        self._specific_model['fixture'] = widget.get_active()

    def toggle_calibration(self, widget, *args, **kwargs):

        view = self.get_view().get_stage()
        val = widget.get_active()
        sm = self._specific_model
        sm['log-only-calibration'] = val
        if val:
            sm['log-previous-file'] = None

        view.set_is_calibration(val)

    def load_previous_log_file(self, widget, view):

        log_file = view_analysis.select_file(
            self._model['analysis-stage-log-title'],
            multiple_files=False,
            file_filter=
            self._model['analysis-stage-log-save-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        if len(log_file) > 0:

            log_file = log_file[0]

            try:

                fs = open(log_file, 'r')

            except:

                return

            headers = fs.readline().strip().split("\t")
            fs.close()

            headers = [h[1:-1] for h in headers]
            interests = [h.split(": ") for h in headers if ":" in h]
            compartments = {i[0] for i in interests}
            measures = {i[1] for i in interests}

            self._specific_model['log-interests'] = [list(compartments),
                list(measures)]

            self._specific_model['log-previous-file'] = log_file

            view.set_interests_from_model()
            view.set_lock_selection_of_interests(True)
            view.set_previous_log_file(log_file)

        else:

            view.set_lock_selection_of_interests(False)
            view.set_previous_log_file("")

    def set_new_images(self, widget, view, *args, **kwargs):

        image_list = view_analysis.select_file(
            self._model['analysis-stage-image-selection-file-dialogue-title'],
            multiple_files=True,
            file_filter=
            self._model['analysis-stage-image-selection-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        treemodel = self._specific_model['images-list-model']

        if len(treemodel) == 0:

            previous_paths = list()

        else:

            previous_paths = [p[0] for p in treemodel if p[0] is not None]

        for im in image_list:

            if im not in previous_paths:

                treemodel.append((im,))

        self._view.get_top().set_allow_next(len(treemodel) > 0)

    def log_compartments(self, widget):

        rows = widget.get_selected_rows()[1]
        self._specific_model['log-interests'][0] = \
            [self._specific_model['log-compartments-default'][r[0]]
            for r in rows]
        
    def log_measures(self, widget):

        rows = widget.get_selected_rows()[1]
        self._specific_model['log-interests'][1] = \
            [self._specific_model['log-measures-default'][r[0]]
            for r in rows]

    def handle_mpl_keypress(self, event):

        if event.key == "delete":

            if len(self._specific_model['plate-coords']) > 0:

                del self._specific_model['plate-coords'][-1]
                self._view.get_stage().remove_patch()  
 

    def handle_keypress(self, widget, event):

        sm = self._specific_model

        if view_analysis.gtk.gdk.keyval_name(event.keyval) == "Delete":

            if sm['stage'] == 'image-selection' or \
                sm['stage'] == 'manual-calibration':

                self._view.get_stage().delete_selection()

    def remove_selection(self, *stuff):

        sm = self._specific_model

        if sm['stage'] == 'manual-calibration':

            mcv = sm['manual-calibration-values']

            val = stuff[0]

            for i in xrange(len(mcv[-1])):

                if val == str(mcv[-1][i]):

                    del sm['manual-calibration-positions'][-1][i]
                    del mcv[-1][i]

                    if len(mcv[-1]) == len(mcv[0]) and len(mcv[-1]) >= 1:

                        self._view.get_top().set_allow_next(True)

                    else:

                        self._view.get_top().set_allow_next(False)

                    return i

        return -1

    def mouse_button_press(self, event, *args, **kwargs):

        if event.xdata is None or event.ydata is None:

            return None

        pos = (event.xdata, event.ydata)
        sm = self._specific_model

        if event.button == 1:

            if sm['stage'] == 'manual-calibration':

                if sm['manual-calibration-positions'] is None:
                    sm['manual-calibration-positions'] = list()

                mc = sm['manual-calibration-positions']

                if len(mc) == sm['image']:

                    mc.append(list())

                if len(mc[-1]) > 0 and len(mc[-1][-1]) == 1:

                    mc[-1][-1][0] = pos

                else:

                    mc[-1].append([pos])

                self._view.get_stage().place_patch_origin(pos)

            elif self._specific_model['stage'] == 'sectioning':

                pc = self._specific_model['plate-coords']

                if len(pc) > 0 and len(pc[-1]) == 1:

                    pc[-1] = pos

                else:

                    pc.append([pos])

                self._view.get_stage().place_patch_origin(pos)

            elif sm['stage'] == 'plate':

                if self._get_inside_selection(pos):

                    sm['selection-move-source'] = pos

                else:

                    if sm['lock-selection'] is not None:
 
                        self.set_selection(pos=pos)
                        self._view.get_stage().move_patch_origin(pos)

                    else:

                        sm['selection-move-source'] = None
                        self.set_selection(pos=pos, wh=(0, 0))
                        specific_view = self._view.get_stage()
                        specific_view.move_patch_origin(pos)
                        specific_view.move_patch_target(0, 0)
                        sm['selection-drawing'] = True

    def _get_inside_selection(self, pos):

        if self._specific_model['selection-origin'] is None or \
            self._specific_model['selection-size'] is None:

            return False

        s_origin = self._specific_model['selection-origin']
        s_target = [p + s for p, s in zip(s_origin,
            self._specific_model['selection-size'])]

        for d in xrange(2):

            if not(s_origin[d] <= pos[d] <= s_target[d]):

                return False

        return True

    def set_selection(self, pos=False, wh=False):

        if pos != False:
            self._specific_model['selection-origin'] = pos

        if wh != False:
            self._specific_model['selection-size'] = wh

    def mouse_button_release(self, event, *args, **kwargs):

        pos = (event.xdata, event.ydata)

        if event.button == 1:

            if self._specific_model['stage'] == 'manual-calibration':

                mc = self._specific_model['manual-calibration-positions'][-1]

                if event.xdata is None or event.ydata is None:

                    if len(mc[-1]) == 1:

                        del mc[-1]
                        self._view.get_stage().remove_patch()

                    return None

                if len(mc[-1]) == 1:

                    origin_pos = mc[-1][0]
                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]
                    mc[-1] = zip(*map(sorted, zip(origin_pos, pos)))

                    self._view.get_stage().move_patch_target(w, h)

                    self.set_manual_calibration_value(mc[-1])

            elif self._specific_model['stage'] == 'sectioning':

                pc = self._specific_model['plate-coords']

                if event.xdata is None or event.ydata is None:

                    if len(pc[-1]) == 1:

                        del pc[-1]
                        self._view.get_stage().remove_patch()

                    self._view.get_stage().set_focus_on_im()
                    return None

                if len(pc[-1]) == 1:

                    origin_pos = pc[-1][0]
                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]
                    pc[-1] = zip(*map(sorted, zip(origin_pos, pos)))

                    self._view.get_stage().move_patch_target(w, h)

                    if len(pc) > 0:

                        self._view.get_top().set_allow_next(True)

                    else:

                        self._view.get_top().set_allow_next(False)

                    self._view.get_stage().set_focus_on_im()

            elif self._specific_model['stage'] == 'plate':

                sm = self._specific_model
                view = self._view.get_stage()
                view.set_image_sensitivity(False)
                origin_pos = sm['selection-origin']

                if sm['selection-move-source'] is not None:

                    self.set_selection(pos=self._get_new_selection_origin(pos))
                    view.move_patch_origin(sm['selection-origin'])

                elif sm['lock-selection'] is None and origin_pos is not None \
                        and None not in origin_pos and None not in pos:

                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]

                    view.move_patch_target(w, h)
                    self.set_selection(wh=(w, h))
                    sm['selection-drawing'] = False

                sm['selection-move-source'] = None
                pos1 = sm['selection-origin']
                wh = view.get_selection_size()
                pos2 = [p + s for p, s in zip(pos1, wh)]

                sm['plate-section-im-array'] = \
                    sm['plate-im-array'][pos1[1]:pos2[1], pos1[0]:pos2[0]]

                sm['plate-section-grid-cell'] = \
                            a_wrapper.get_grid_cell_from_array(
                            sm['plate-section-im-array'], center=None,
                            radius=None,
                            invoke_transform=sm['plate-is-normed'])

                sm['plate-section-features'] = \
                    sm['plate-section-grid-cell'].get_analysis()

                view.set_section_image()
                view.set_analysis_image()

                strain = self._log.get_suggested_strain_name(pos1)
                if strain is not None:
                    view.set_strain(strain)

                self.set_allow_logging()
                view.set_image_sensitivity(True)

    def set_allow_logging(self):

        sm = self._specific_model

        self._view.get_stage().set_allow_logging(
            not(sm['plate-section-features'] is None)
            and self._log.get_all_meta_filled())
 
    def _get_new_selection_origin(self, pos):

        sm = self._specific_model
        sel_move = [n - o for n, o in zip(pos, sm['selection-move-source'])]

        new_origin = [o + m for o, m in zip(sm['selection-origin'], sel_move)]

        return new_origin

    def set_manual_calibration_value(self, coords):

        if self._specific_model['manual-calibration-values'] is None:

            self._specific_model['manual-calibration-values'] = list()

        mcv = self._specific_model['manual-calibration-values']

        if len(mcv) == self._specific_model['image']:

            mcv.append(list())

        mcv[-1]. append(
            self._specific_model['image-array'][coords[0][1]: coords[1][1],
            coords[0][0]: coords[1][0]].mean())

        self._view.get_stage().add_measure(mcv[-1][-1])

        if len(mcv[-1]) == len(mcv[0]) and len(mcv[-1]) >= 1:

            self._view.get_top().set_allow_next(True)

        else:

            self._view.get_top().set_allow_next(False)

    def mouse_move(self, event, *args, **kwargs):

        sm = self._specific_model
        pos = (event.xdata, event.ydata)

        if event.xdata is None or event.ydata is None:

            return None

        if sm['stage'] == 'manual-calibration':

            mc = sm['manual-calibration-positions']

            if mc is not None and mc[-1] is not None and len(mc[-1]) > 0 \
                and len(mc[-1][-1]) == 1:
                
                origin_pos = mc[-1][-1][0]
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]
                self._view.get_stage().move_patch_target(w, h)

        elif sm['stage'] == 'sectioning':

            pc = self._specific_model['plate-coords']

            if len(pc) > 0 and len(pc[-1]) == 1:
                
                origin_pos = pc[-1][0]
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]
                self._view.get_stage().move_patch_target(w, h)

        elif sm['stage'] == 'plate':

            if sm['selection-move-source'] is not None:

                self._view.get_stage().move_patch_origin(
                        self._get_new_selection_origin(pos))

            elif sm['lock-selection'] is None and \
                sm['selection-origin'] is not None \
                and sm['selection-drawing'] == True:

                origin_pos = sm['selection-origin']
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]

                self._view.get_stage().move_patch_target(w, h)

    def set_cell(self, widget, type_of_value):

        stage = self._view.get_stage()

        wh = list(stage.get_selection_size())
        if type_of_value ==  "height":
            try:
                h = int(widget.get_text())
            except:
                return None
        else:
            h = wh[1]

        if type_of_value ==  "width":
            try:
                w = int(widget.get_text())
            except:
                return None
        else:
            w = wh[0]

        self.set_selection(wh=(w, h))
        stage.move_patch_target(w, h)

    def set_selection_lock(self, widget):

        if widget.get_active():

            self._specific_model['lock-selection'] = \
                self._view.get_stage().get_selection_size()

        else:

            self._specific_model['lock-selection'] = None

        self._view.get_stage().set_allow_selection_size_change(
            self._specific_model['lock-selection'] == None)

    def set_in_log(self, widget, key):

        self._log.set(key, widget)
        self.set_allow_logging()


class Analysis_Transparency_Controller(Analysis_Image_Controller):

    def __init__(self, parent, view=None, model=None, logger=None):

        super(Analysis_Transparency_Controller, self).__init__(
                parent, view=view, model=model, logger=logger)

    def build_blank_specific_model(self):

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_transparency))


class Analysis_Project_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None, logger=None):

        super(Analysis_Project_Controller, self).__init__(
                parent, view=view, model=model, logger=logger) 

        self.build_blank_specific_model()


    def set_abort(self, *args):

        self._parent.set_analysis_stage(None, "about")
        
    def build_blank_specific_model(self):

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_project))

    def start(self, *args, **kwargs):

        sm = self._specific_model
        tc = self.get_top_controller()

        tc.paths.experiment_analysis_relative
        analysis_log = open(os.sep.join(sm['analysis-project-log_file_dir'],
            sm['analysis-project-output-path'],
            tc.paths.experiment_analysis_file_name) , 'w')

        a_dict = tc.config.get_default_analysis_query()
        a_dict['-i'] = sm['analysis-project-log_file']
        a_dict['-o'] = sm['analysis-project-output-path']

        analysis_query = [tc.paths.analysis]

        for a_flag, a_val in a_dict.items():

            analysis_query += [a_flag, a_val]

        if sm['analysis-project-pinnings-active'] != 'file':
            pm = ""
            for plate in sm['analysis-project-pinnings']:
                pm += str(plate).replace(" ","") + ":"
            pm = pm[:-1]
            analysis_query += ["-m", pm]

        """Functionality not implemented:
        if self._watch_colony is not None:
            analysis_query += ["-w", self._watch_colony]
        if self._supress_other is True: 
            analysis_query += ["-s", "True"]
        """
        proc = Popen(map(str, analysis_query), 
            stdout=analysis_log, shell=False)

        pid = proc.pid

        self.get_top_controller().add_subprocess(proc, 'analysis', pid=pid,
            stdout=analysis_log, psm=sm,
            proc_name=sm['analysis-project-log_file'])

    def set_log_file(self, *args, **kwargs):

        log_files = view_analysis.select_file(
            self._model['analysis-stage-project-select-log-file-dialog'],
            multiple_files=False, file_filter=
            self._model['analysis-stage-project-select-log-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        if len(log_files) > 0:

            sm = self._specific_model
            view = self._view.get_stage()

            sm['analysis-project-log_file'] = log_files[0]
            sm['analysis-project-log_file_dir'] = \
                os.sep.join(log_files[0].split(os.sep)[: -1])

            sm['analysis-project-pinnings-active'] = 'file'

            meta_data, images = resource_project_log.get_log_file(
                                log_files[0])

            if 'Pinning Matrices' in meta_data:

                pinning_matrices = meta_data['Pinning Matrices']

            else:

                plates=resource_project_log.get_number_of_plates(
                                meta_data=meta_data, images=images)
                if plates > 0:

                    pinning_matrices = [None] * plates

                else:

                    pinning_matrices = None

            sm['analysis-project-pinnings-from-file'] = pinning_matrices
            sm['analysis-project-pinnings'] = copy.copy(pinning_matrices)

            view.set_log_file()

            view.set_log_file_data(
                meta_data['Prefix'], meta_data['Description'],
                str(len(images)))

            view.set_pinning(pinning_matrices)

            self.set_output_dupe()

        self.set_ready_to_run()

    def set_output(self, widget, view, event):

        output_path = widget.get_text()
        
        output_path = resource_os.get_valid_relative_dir(output_path,
                "")

        sm = self._specific_model

        if event == "exit" and output_path == "":

            output_path = sm['analysis-project-output-default']

        sm['analysis-project-output-path'] = output_path

        if output_path != widget.get_text():

            view.correct_output_path(output_path)

        self.set_output_dupe(output_path, view)
        self.set_ready_to_run()

    def set_output_dupe(self, rel_path=None, view=None):

        sm = self._specific_model

        if view is None:
            view = self._view.get_stage()

        if rel_path is None:
            rel_path = sm['analysis-project-output-path']

        full_path = sm['analysis-project-log_file_dir'] + \
            os.sep + rel_path

        view.set_output_warning(os.path.isdir(full_path))

    def toggle_set_pinning(self, widget, view):

        sm = self._specific_model

        if widget.get_active():

            sm['analysis-project-pinnings-active'] = 'file'
            view.set_pinning(
                sm['analysis-project-pinnings-from-file'],
                sensitive=False)

        else:

            sm['analysis-project-pinnings-active'] = 'gui'
            view.set_pinning(
                sm['analysis-project-pinnings'],
                sensitive=True)

        self.set_ready_to_run()
            
    def set_pinning(self, widget, plate, *args, **kwargs):

        sm = self._specific_model
        m = self._model

        pinning_txt = widget.get_active_text()

        if pinning_txt in m['pinning-matrices']:
            pinning = m['pinning-matrices'][pinning_txt]
        else:
            pinning = None

        sm['analysis-project-pinnings'][plate - 1] = pinning

        self.set_ready_to_run()

    def set_ready_to_run(self):

        sm = self._specific_model

        if sm['analysis-project-pinnings-active'] == 'file':

            sm_key = 'analysis-project-pinnings-from-file'

        else:

            sm_key = 'analysis-project-pinnings'

        plates_ok = sum([p != None for p in sm[sm_key]]) > 0

        file_loaded = sm['analysis-project-log_file'] != ""

        self._view.get_top().set_allow_next(file_loaded and plates_ok)


class Analysis_Log_Controller(controller_generic.Controller):

    def __init__(self, parent, general_model, parent_model,
            logger=None):

        model = model_analysis.copy_model(model_analysis.specific_log_book)
        self._parent_model = parent_model
        self._general_model = general_model
        self._look_up_coords = list()
        self._look_up_names = list()

        super(Analysis_Log_Controller, self).__init__( 
            parent, model=model,
            view=view_analysis.Analysis_Stage_Log(self, general_model,
            model, parent_model), logger=logger)

        if self._parent_model['log-only-calibration']:
            self._model['calibration-measures'] = True

        if self._parent_model['log-previous-file'] is not None:
            self._load_previous_file_contents()


    def _get_default_view(self):

        view = view_analysis.Analysis_Stage_Log(self, self._general_model,
            self._model, self._parent_model)

        return view

    def _load_previous_file_contents(self):

        log_file = self._parent_model['log-previous-file']
        try:

            fs = open(log_file, 'r')

        except:

            return False

        measures = self._model['measures']
        headers = fs.readline().strip().split("\t")

        for data_row in fs:

            data_row = data_row.strip().replace("\t",", ")
            data = eval("[{0}]".format(data_row))
            for i, d in enumerate(data):
                if type(d) == types.StringType and len(d) > 0 and \
                            d[0] in ('[', '(') and d[-1] in (']', ')'):

                    try:
                        data[i] = eval(d)
                    except:
                        pass

            measures.append(data)
            self._view.add_data_row(data)


        fs.close()

        return True

    def get_all_meta_filled(self):

        m = self._model
        pm = self._parent_model

        try:

            all_ok = m['plate-names'][pm['image']] is not None and \
                len(m['plate-names'][pm['image']]) == pm['plate'] + 1 and \
                m['current-strain'] is not None

        except:

            return False

        if all_ok and m['calibration-measures']:

            if m['indie-count'] is None:

                all_ok = False

        return all_ok

    def _set_look_up(self, m):

        self._look_up_coords.append(m[4][0])
        self._look_up_names.append(m[5])

    def _set_look_up_strains(self, plate_index=None):

        self._look_up_coords = list()
        self._look_up_names = list()

        measures = self._model['measures']
        
        if plate_index is not None:

            for m in measures:

                if m[2] == plate_index:

                    self._set_look_up(m)

        if len(self._look_up_coords) == 0:

            for m in measures:

                self._set_look_up(m)

        self._look_up_coords = np.array(self._look_up_coords)

    def get_suggested_strain_name(self, coords):

        if len(self._look_up_names) == 0 :

            self._set_look_up_strains(self._parent_model['plate'])

        if self._look_up_coords.size > 0:

            strain_pos = ((self._look_up_coords - coords)**2).sum(1).argmin()

            if len(self._look_up_names) > strain_pos >= 0:
                strain = self._look_up_names[strain_pos]
                return strain

        return None

    def get_suggested_plate_name(self, index):

        measures = self._model['measures']

        plate_name = None

        for m in measures:

            if m[2] == index:

                plate_name = m[3] 

        self._set_look_up_strains(index)

        return plate_name

    def set(self, key, item):

        if key == 'plate':

            image = self._parent_model['image']
            plate = self._parent_model['plate']

            if len(self._model['images']) <= image:
                self._model['images'].append(
                    self._parent_model['images-list-model'][image][0])
                self._model['plate-names'].append(list())

            if len(self._model['plate-names'][image]) <= plate:

                self._model['plate-names'][image].append(item.get_text())

            else:

                self._model['plate-names'][image][plate] = item.get_text()

        elif key == 'strain':

            self._model['current-strain'] = item.get_text()

        elif key == 'indie-count':

            try:
                self._model['indie-count'] = float(item.get_text())

            except:

                self._model['indie-count'] = None

        elif key == 'measures':

            pm = self._parent_model
            m = self._model

            #META INFO
            measures = [m['images'][-1],  # pos 0, image-path
                pm['plate-coords'][pm['plate']],  # pos 1 plate coordinates
                pm['plate'],  # pos 2 plate index
                m['plate-names'][pm['image']][pm['plate']],  # pos 3 plate name
                (pm['selection-origin'], pm['selection-size']),  # pos 4 selection coords
                m['current-strain']]  # 5 strain name

            if m['calibration-measures']:

                #SHOULD BE IN RESOURCE
                blob = pm['plate-section-grid-cell'].get_item(
                    "blob").filter_array
                bg = pm['plate-section-grid-cell'].get_item(
                    "background").filter_array
                bg_mean = pm['plate-section-im-array'][bg].mean()
                blob_pixels = pm['plate-section-im-array'][blob] - bg_mean
                blob_pixels[blob_pixels<0] = 0
                k = np.unique(blob_pixels)
                c = list()
                for v in k:
                    c.append(blob_pixels[blob_pixels==v].size)

                measures += [m['indie-count'], list(k), c]

            else:

                features = pm['plate-section-features']

                for compartment in pm['log-interests'][0]:

                    if compartment in features.keys():

                        c = features[compartment]

                        for measure in pm['log-interests'][1]:

                            if measure in c.keys():

                                measures.append(c[measure])

                            else:

                                measures.append(None)

            m['measures'].append(measures)

            self._set_look_up_strains(plate_index=pm['plate'])

            self._view.add_data_row(measures)

        else:

            err = Unknown_Log_Request(
                "The key '{0}' not recognized (item {1} lost)".format(
                key, item))

            raise err

    def save_data(self, widget):

        file_name = view_analysis.save_file(
            self._general_model['analysis-stage-log-save-dialog'],
            multiple_files=False,
            file_filter=self._general_model['analysis-stage-log-save-file-filter'],
            start_in=self.get_top_controller().paths.experiment_root)

        file_saved = False

        if len(file_name) > 0:

            file_name = file_name[0]

            #Check so endin was filled in
            ext_str = ".csv"
            if ext_str not in file_name or file_name[-4:] != ext_str:
                file_name += ext_str

            try:

                fs = open(file_name, 'r')
                file_exists = True
                fs.close()

            except:

                file_exists = False

            if file_exists:

                file_exists = not(view_analysis.overwrite(
                    self._general_model['analysis-stage-log-overwrite'],
                    file_name, self.get_window()))

            

            if file_exists == False:


                fs = open(file_name, 'w')

                sep = "\t"
                quoute = '"'
                pm = self._parent_model
                m = self._model

                for i, header in enumerate(pm['log-meta-features']):

                    fs.write("{0}{1}{0}{2}".format(quoute, header, sep))

                for i, compartment in enumerate(pm['log-interests'][0]):

                    for j, measure in enumerate(pm['log-interests'][1]):

                        fs.write("{0}{1}: {2}{0}".format(quoute,
                            compartment, measure) )

                        if j + 1 != len(pm['log-interests'][1]):

                            fs.write(sep)

                    if i + 1 != len(pm['log-interests'][0]):

                        fs.write(sep)

                fs.write("\n\r")
 
                for i, measure in enumerate(m['measures']):

                    for j, val in enumerate(measure):

                        if type(val) == types.IntType or type(val) == types.FloatType:

                            fs.write(str(val))

                        else:

                            fs.write("{0}{1}{0}".format(quoute, val))

                        if j + 1 != len(measure):

                            fs.write(sep)

                    if i + 1  != len(m['measures']):

                        fs.write("\n\r")

                fs.close()

                file_saved = True
                self._parent.set_saved()

                view_analysis.dialog(self.get_window(),
                    self._general_model['analysis-stage-log-saved'],
                    d_type='info')

        if file_saved == False:

            view_analysis.dialog(self.get_window(),
                self._general_model['analysis-stage-log-not-saved'],
                d_type='warning')

    def handle_keypress(self, widget, event):

        if view_analysis.gtk.gdk.keyval_name(event.keyval) == "Delete":

            self._view.delete_selection()  

    def remove_selection(self, *stuff):

        m = self._model

        im_path = stuff[0]
        plate = int(stuff[1])
        pos = eval(stuff[2])

        for i in xrange(len(m['measures'])):

            if im_path == m['measures'][i][0] and \
                plate == m['measures'][i][2] and \
                pos == m['measures'][i][4]:

                del m['measures'][i]

                return i

        return -1
