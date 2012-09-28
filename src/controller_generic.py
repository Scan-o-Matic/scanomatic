

class Controller(object):

    def __init__(self, model=None, view=None, specific_model=None):

        #MODEL SHOULD BE SET BEFORE VIEW!
        self.set_model(model)
        self.set_specific_model(specific_model)
        self.set_view(view)

    def set_view(self, view=None):

        if view is None:

            view = self._get_default_view()

        self._view = view

    def get_view(self):

        return self._view

    def _get_default_view(self):

        return None

    def set_model(self, model=None):

        if model is None:

            model = self._get_default_model()

        self._model = model

    def _get_default_model(self):

        return None

    def get_model(self):

        return self._model

    def set_specific_model(self, specific_model):

        self._specific_model = specific_model

    def get_specific_model(self):

        if self._specific_model is None:

            self.set_specific_model(dict())

        return self._specific_model

