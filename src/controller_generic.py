

class Controller(object):

    def __init__(self, model=None, view=None):

        #MODEL SHOULD BE SET BEFORE VIEW!
        self.set_model(model)
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
