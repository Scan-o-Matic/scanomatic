import os
import types
import gobject
import threading
import math

import model_analysis
import view_analysis
import controller_generic
import resource_os
import resource_log_reader
import analysis_wrapper as a_wrapper
import resource_fixture
import resource_image

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class Unknown_Log_Request(Exception): pass
class UnDocumented_Error(Exception): pass


class Analysis_Controller(controller_generic.Controller):

    def __init__(self, window):

        self._window = window

        super(Analysis_Controller, self).__init__()

        self.project = Analysis_Project_Controller(view=self._view,
                                model=self._model)

        self.transparency = Analysis_Transparency_Controller(view=self._view,
                                model=self._model)

        self.progress = threading.Lock()
        self.fixture = None

    def _get_default_view(self):

        return view_analysis.Analysis(self, self._model)

    def _get_default_model(self):

        return model_analysis.model

    def _callback(self, user_data):

        if user_data is not None:

            if user_data['view-data'] is None and 'cb' not in user_data:

                user_data['cb'] = 9

            if user_data['thread'].is_alive() == False:

                user_data['view-function'](user_data['view-complete'])
                print "Done!"
                self.progress.release()
                user_data['view'].run_release()

                return False

            else:

                if user_data['view-data'] is None:

                    user_data['cb'] += 1
                    user_data['view-function'](1 - math.exp(-0.01 * user_data['cb']))

                else:

                    user_data['view-function'](user_data['view-data'])

        if user_data is None:

            print "LOST!"
            return False

        gobject.timeout_add(250, self._callback, user_data)
        return None

    def get_available_fixtures(self):

        directory = self._model['fixtures-path']
        extension = ".config"
        list_fixtures = map(lambda x: x.split(extension,1)[0], [file for file\
            in os.listdir(directory) if file.lower().endswith(extension)])

        return sorted(list_fixtures)

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

                    specific_model['image'] += 1

                    if specific_model['image'] >= len(specific_model['images-list-model']):

                        raise Bad_Stage_Call("Image position overflow")

                    if specific_model['fixture']:

                        specific_model['stage'] = 'auto-calibration'
                        specific_model['plate'] = -1

                        model['fixtures'] = self.get_available_fixtures()

                        view.set_top(
                            view_analysis.Analysis_Top_Auto_Norm_and_Section(
                            self, model,
                            specific_model,
                            self.transparency))

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
                    self._window))

                specific_model['plate'] = -1

            elif stage_call == "plate":

                specific_model = args[1]
                specific_model['plate'] += 1
                specific_model['stage'] = 'plate'

                if self.transparency._log is None:
                    self.transparency._log = Analysis_Log_Controller(
                        self._model, specific_model, self._window)

                else:

                    self.transparency._log.set_view()
                    self.transparency._log._model['current-strain'] = None

                if specific_model['plate'] < len(specific_model['plate-coords']):

                    coords = specific_model['plate-coords'][specific_model['plate']]
                    image = specific_model['image']
                    
                    if image in specific_model['auto-calibration-values']:

                        image_transpose = resource_image.Image_Transpose()
                        image_transpose.set_matrix(
                            gs_values=specific_model['auto-calibration-values'][image],
                            gs_indices=specific_model['auto-calibration-indices'][image])

                        specific_model['plate-im-array'] = image_transpose.get_transposed_im(
                            specific_model['image-array'][
                            coords[0][1]: coords[1][1],
                            coords[0][0]: coords[1][0]])

                    else:
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
                    
                    view.get_stage().run_lock_select_check()
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

    def execute_fixture(self, widget, data):

        view, specific_model = data
        view.run_lock()

        self.fixture = resource_fixture.Fixture_Settings(
            self._model['fixtures-path'],
            fixture=specific_model["fixture-name"],
            image=specific_model['images-list-model'][
            specific_model['image']][0],
            markings=-1)

        thread = threading.Thread(target=self.fixture.threaded)

        thread.start()
        self.progress.acquire()

        gobject.timeout_add(250, self._callback, {
            'view': view,
            'view-function': view.set_progress,
            'view-data':None,
            'view-complete': 1.0,
            'thread': thread})

    def get_previously_detected(self, view, specific_model):

        image=specific_model['images-list-model'][
                specific_model['image']][0]

        im_dir = os.sep.join(image.split(os.sep)[:-1])

        extension = ".log"

        log_files = [im_dir + os.sep + f for f in os.listdir(im_dir) 
                        if f.lower().endswith(extension)]

        data = None

        for f in log_files:

            data = resource_log_reader.get_im_data(f, image)

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

                data_sorted =  zip(*map(sorted, zip(*data['plate_{0}_area'.format(i)])))
                scale_sorted = [[4*p[1], 4*p[0]] for p in data_sorted]
                plate_coords[i] = scale_sorted
                i += 1

            specific_model['plate-coords'] = plate_coords

            self.set_auto_grayscale(data['grayscale_values'],
                data[gs_i_key])

            self.get_view().get_top().set_allow_next(True)

    def set_fixture(self, view, fixture_name, specific_model):

        specific_model['fixture-name'] = fixture_name

    def set_auto_grayscale(self, vals, indices):

        sm = self._specific_model

        sm['auto-calibration-values'][sm['image']] = vals
        sm['auto-calibration-indices'][sm['image']] = indices
            
    def set_grayscale(self, view):

        if self.fixture is not None:

            grayscale_im = self.fixture['image'].get_subsection(
                self.fixture['current'].get("grayscale_area"))

            ag = resource_image.Analyse_Grayscale(target_type="Kodak",
                image=grayscale_im, scale_factor=1.0, dpi=600)

            print ag.get_grayscale()

        else:

            raise UnDocumented_Error()

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
                self.set_allow_logging()

    def set_allow_logging(self):

        sm = self._specific_model

        print "Selection exists:", not(sm['plate-section-features'] is None)

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
        self.set_allow_logging()


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

    def __init__(self, general_model, parent_model, window):

        model = model_analysis.copy_model(model_analysis.specific_log_book)
        self._parent_model = parent_model
        self._general_model = general_model
        self._window = window

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

        print

        print 'plate-names', m['plate-names']
        print 'image', pm['image']
        print 'plate', pm['plate']
        print 'current-strain', m['current-strain']

        print
        try:

            all_ok = m['plate-names'][pm['image']] is not None and \
                len(m['plate-names'][pm['image']]) == pm['plate'] + 1 and \
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

    def save_data(self, widget):

        file_name = view_analysis.save_file(
            self._general_model['analysis-stage-log-save-dialog'],
            multiple_files=False,
            file_filter=self._general_model['analysis-stage-log-save-file-filter'])
            
        file_saved = False

        if len(file_name) > 0:

            file_name = file_name[0]

            try:

                fs = open(file_name, 'r')
                file_exists = True
                fs.close()

            except:

                file_exists = False

            if file_exists:

                file_exists = not(view_analysis.overwrite(
                    self._general_model['analysis-stage-log-overwrite'],
                    file_name, self._window))

            

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

                view_analysis.dialog(self._window,
                    self._general_model['analysis-stage-log-saved'],
                    d_type='info')

        if file_saved == False:

            view_analysis.dialog(self._window,
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
