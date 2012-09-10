import os

import model_analysis
import view_analysis
import controller_generic
import resource_os
import analysis_wrapper as a_wrapper

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class Unknown_Log_Request(Exception): pass

class Analysis_Controller(controller_generic.Controller):

    def __init__(self, window):

        self._window = window

        super(Analysis_Controller, self).__init__()

        self.project = Analysis_Project_Controller(view=self._view,
                                model=self._model)

        self.transparency = Analysis_Transparency_Controller(view=self._view,
                                model=self._model)

    def _get_default_view(self):

        return view_analysis.Analysis(self, self._model)

    def _get_default_model(self):

        return model_analysis.model

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

                view.set_top(view_analysis.Analysis_Top_Project(
                                                        self, model))

                view.set_stage(view_analysis.Analysis_Stage_Project(
                                                        self, model))

            elif stage_call == "transparency":

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


                if specific_model['mode'] == 'transparency':

                    if specific_model['fixture']:

                        raise Not_Yet_Implemented((stage_call, ('fixture', specific_model['fixture'])))

                    else:

                        specific_model['stage'] = 'manual-calibration'
                        specific_model['image'] += 1

                        if specific_model['image'] >= len(specific_model['images-list-model']):

                            raise Bad_Stage_Call("Image position overflow")

                        else:

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

                specific_model = args[1]
                specific_model['stage'] = 'sectioning'

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
                    self._window))

            elif stage_call == "plate":

                specific_model = args[1]
                specific_model['plate'] += 1
                specific_model['stage'] = 'plate'

                if self.transparency._log is None:
                    self.transparency._log = Analysis_Log_Controller(
                        self._model, specific_model)

                else:

                    self.transparency._log.set_view()

                if specific_model['plate'] < len(specific_model['plate-coords']):

                    coords = specific_model['plate-coords'][specific_model['plate']]

                    specific_model['plate-im-array'] = \
                        specific_model['image-array'][
                        coords[0][1]: coords[1][1],
                        coords[0][0]: coords[1][0]]

                    view.set_top(
                        view_analysis.Analysis_Top_Image_Plate(
                        self, model,
                        specific_model,
                        self.transparency))

                    view.get_top().set_allow_next(True)

                    view.set_stage(
                        view_analysis.Analysis_Stage_Image_Plate(
                        self, model,
                        specific_model,
                        self.transparency))
                    
            else:

                raise Bad_Stage_Call(stage_call)


class Analysis_Image_Controller(controller_generic.Controller):

    def __init__(self, view=None, model=None):

        super(Analysis_Image_Controller, self).__init__(view=view,
                model=model)

        self._specific_model = None
        self._log = None

    def set_specific_model(self, specific_model):

        self._specific_model = specific_model

    def get_specific_model(self):

        if self._specific_model is None:

            self.set_specific_model(dict())

        return self._specific_model

    def set_images_has_fixture(self, widget, *args, **kwargs):

        self._specific_model['fixture'] = widget.get_active()


    def set_new_images(self, widget, view, *args, **kwargs):

        image_list = view_analysis.select_file(
            self._model['analysis-stage-image-selection-file-dialogue-title'],
            multiple_files=True,
            file_filter=
            self._model['analysis-stage-image-selection-file-filter'])

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
        s_target = [p + s for p, s in zip(s_origin,  self._specific_model['selection-size'])]

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

                if sm['selection-move-source'] is not None:

                    self.set_selection(pos=self._get_new_selection_origin(pos))
                    self._view.get_stage().move_patch_origin(sm['selection-origin'])

                elif sm['lock-selection'] is None and sm['selection-origin'] is not None:

                    origin_pos = sm['selection-origin']
                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]

                    self._view.get_stage().move_patch_target(w, h)
                    self.set_selection(wh=(w, h))
                    sm['selection-drawing'] = False

                sm['selection-move-source'] = None
                pos1 = sm['selection-origin']
                wh = self._view.get_stage().get_selection_size()
                pos2 = [p + s for p, s in zip(pos1, wh)]

                sm['plate-section-im-array'] = sm['plate-im-array'][pos1[1]:pos2[1], pos1[0]:pos2[0]]
                sm['plate-section-grid-cell'] = a_wrapper.get_grid_cell_from_array(
                            sm['plate-section-im-array'], center=None,
                            radius=None)

                sm['plate-section-features'] = sm['plate-section-grid-cell'].get_analysis()

                self._view.get_stage().set_section_image()
                self._view.get_stage().set_analysis_image()

                self._view.get_stage().set_allow_logging(not(sm['plate-section-features'] is None)
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

            elif sm['lock-selection'] is None and sm['selection-origin'] is not None \
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


class Analysis_Transparency_Controller(Analysis_Image_Controller):

    def __init__(self, view=None, model=None):

        super(Analysis_Transparency_Controller, self).__init__(view=view,
                model=model)

    def build_blank_specific_model(self):

        self.set_specific_model(model_analysis.copy_model(
            model_analysis.specific_transparency))


class Analysis_Project_Controller(controller_generic.Controller):

    def __init__(self, view=None, model=None):

       super(Analysis_Project_Controller, self).__init__(view=view,
                model=model) 

    def start(self, *args, **kwargs):

        print args, kwargs

    def set_log_file(self, *args, **kwargs):

        print args, kwargs

    def set_output(self, widget, view, event):

        output_path = widget.get_text()
        
        output_path = resource_os.get_valid_relative_dir(output_path,
                "")

        if event == "exit" and output_path == "":

            output_path = self._model['analysis-project-output-default']

        self._model['analysis-project-output-path'] = output_path

        if output_path != widget.get_text():

            view.correct_output_path(output_path)

        self.set_output_dupe(output_path, view)

    def set_output_dupe(self, rel_path, view):

        full_path = self._model['analysis-project-log_file_dir'] + \
            os.sep + rel_path

        view.set_output_warning(os.path.isdir(full_path))

    def toggle_set_pinning(self, widget, view):

        if widget.get_active():

            view.set_pinning(
                self._model['analysis-project-pinnings-from-file'], False)

        else:

            view.set_pinning(
                self._model['analysis-project-pinnings'], True)
            
    def set_pinning(self, widget, view, *args, **kwargs):

        print view, args, kwargs

class Analysis_Log_Controller(controller_generic.Controller):

    def __init__(self, general_model, parent_model):

        model = model_analysis.copy_model(model_analysis.specific_log_book)
        self._parent_model = parent_model
        self._general_model = general_model

        super(Analysis_Log_Controller, self).__init__(model=model,
            view=view_analysis.Analysis_Stage_Log(self, general_model,
            model, parent_model))

    def _get_default_view(self):

        view = view_analysis.Analysis_Stage_Log(self, self._general_model,
            self._model, self._parent_model)

        return view

    def get_all_meta_filled(self):

        m = self._model
        pm = self._parent_model

        """
        print

        print 'plate-names', m['plate-names']
        print 'image', pm['image']
        print 'current-strain', m['current-strain']

        print
        """
        try:

            all_ok = m['plate-names'][pm['image']] is not None and \
                len(m['plate-names'][pm['image']]) == pm['image'] + 1 and \
                m['current-strain'] is not None

        except:

            return False

        return all_ok

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

        elif key == 'measures':

            pm = self._parent_model
            m = self._model

            #META INFO
            measures = [m['images'][-1], pm['plate-coords'][pm['plate']],
                pm['plate'], m['plate-names'][pm['image']][pm['plate']],
                (pm['selection-origin'], pm['selection-size']),
                m['current-strain']]

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


            self._view.add_data_row(measures)

        else:

            raise Unknown_Log_Request("The key '{0}' not recognized (item {1} lost)".format(key, item))

    def handle_keypress(self, widget, event):

        if view_analysis.gtk.gdk.keyval_name(event.keyval) == "Delete":

            self._view.delete_selection()  

    def remove_selection(self, *stuff):

        m = self._model

        im_path = stuff[0]
        plate = stuff[1]
        pos = stuff[2]

        for i in xrange(len(m['measures'])):

            if im_path == m['measures'][i][0] and \
                plate == m['measures'][i][2] and \
                pos == m['measures'][i][4]:


                del m['measures'][i]

                return i

        return -1
