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
                print specific_model
                print args

                if specific_model['mode'] == 'transparency':

                    if specific_model['fixture']:

                        raise Not_Yet_Implemented((stage_call, ('fixture', specific_model['fixture'])))

                    else:

                        specific_model['image'] += 1

                        if specific_model['image'] >= len(specific_model['images-list-model']):

                            raise Bad_Stage_Call("Image position overflow")

                        else:

                            view.get_top().set_allow_next(False)

                            view.set_stage(
                                view_analysis.Analysis_Stage_Image_Norm_Manual(
                                self, model,
                                self.transparency.get_specific_model(),
                                self.transparency))

                elif specific_model['mode'] == 'colour':

                    raise Not_Yet_Implemented((stage_call, specific_model['mode']))

                else:

                    raise Bad_Stage_Call(stage_call)

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

        print len(treemodel)

        self._view.get_top().set_allow_next(len(treemodel) > 0)

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
