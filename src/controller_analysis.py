
import model_analysis
import view_analysis
import controller_generic

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass


class Analysis_Controller(controller_generic.Controller):

    def __init__(self):

        super(Analysis_Controller, self).__init__()
        self.project = Analysis_Project_Controller(view=self._view,
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

            else:

                raise Bad_Stage_Call()


class Analysis_Project_Controller(controller_generic.Controller):

    def __init__(self, view=None, model=None):

       super(Analysis_Project_Controller, self).__init__(view, model) 


    def start(self, *args, **kwargs):

        print args, kwargs
