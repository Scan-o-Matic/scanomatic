import os

import model_analysis
import view_analysis
import controller_generic
import resource_os

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass

class Analysis_Controller(controller_generic.Controller):

    def __init__(self):

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
                    self.transparency))


            else:

                raise Bad_Stage_Call(stage_call)


class Analysis_Image_Controller(controller_generic.Controller):

    def __init__(self, view=None, model=None):

        super(Analysis_Image_Controller, self).__init__(view=view,
                model=model)

        self._specific_model = None

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

    def mouse_button_press(self, event, *args, **kwargs):

        if event.xdata is None or event.ydata is None:

            return None

        if event.button == 1:

            if self._specific_model['stage'] == 'manual-calibration':

                if self._specific_model['manual-calibration-positions'] is None:
                    self._specific_model['manual-calibration-positions'] = list()

                mc = self._specific_model['manual-calibration-positions']

                if len(mc) == self._specific_model['image']:

                    mc.append(list())

                pos = (event.xdata, event.ydata)

                if len(mc[-1]) > 0 and len(mc[-1][-1]) == 1:

                    mc[-1][-1][0] = pos

                else:

                    mc[-1].append([pos])

                self._view.get_stage().place_patch_origin(pos)

            elif self._specific_model['stage'] == 'sectioning':

                pc = self._specific_model['plate-coords']
                pos = (event.xdata, event.ydata)

                if len(pc) > 0 and len(pc[-1]) == 1:

                    pc[-1] = pos

                else:

                    pc.append([pos])

                self._view.get_stage().place_patch_origin(pos)
 
    def mouse_button_release(self, event, *args, **kwargs):

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
                    pos = (event.xdata, event.ydata)
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

                    return None

                if len(pc[-1]) == 1:

                    origin_pos = pc[-1][0]
                    pos = (event.xdata, event.ydata)
                    w = pos[0] - origin_pos[0]
                    h = pos[1] - origin_pos[1]
                    pc[-1] = zip(*map(sorted, zip(origin_pos, pos)))

                    self._view.get_stage().move_patch_target(w, h)

                    self.set_manual_calibration_value(pc[-1])

                    if len(pc) > 0:

                        self._view.get_top().set_allow_next(True)

                    else:

                        self._view.get_top().set_allow_next(False)
 
    def set_manual_calibration_value(self, coords):

        if self._specific_model['manual-calibration-values'] is None:

            self._specific_model['manual-calibration-values'] = list()

        mcv = self._specific_model['manual-calibration-values']

        if len(mcv) == self._specific_model['image']:

            mcv.append(list())

        mcv[-1]. append(
            self._specific_model['image-array'][coords[0][0]: coords[1][0],
            coords[0][1]: coords[1][1]].mean())

        self._view.get_stage().add_measure(mcv[-1][-1])

        if len(mcv[-1]) == len(mcv[0]) and len(mcv[-1]) >= 2:

            self._view.get_top().set_allow_next(True)

        else:

            self._view.get_top().set_allow_next(False)

    def mouse_move(self, event, *args, **kwargs):

        if event.xdata is None or event.ydata is None:

            return None

        if self._specific_model['stage'] == 'manual-calibration':

            mc = self._specific_model['manual-calibration-positions']

            if mc is not None and mc[-1] is not None and len(mc[-1]) > 0 \
                and len(mc[-1][-1]) == 1:
                
                origin_pos = mc[-1][-1][0]
                pos = (event.xdata, event.ydata)
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]
                self._view.get_stage().move_patch_target(w, h)

        elif self._specific_model['stage'] == 'sectioning':

            pc = self._specific_model['plate-coords']

            if len(pc) > 0 and len(pc[-1]) == 1:
                
                origin_pos = pc[-1][0]
                pos = (event.xdata, event.ydata)
                w = pos[0] - origin_pos[0]
                h = pos[1] - origin_pos[1]
                self._view.get_stage().move_patch_target(w, h)


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
