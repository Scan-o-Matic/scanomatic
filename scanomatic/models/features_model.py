import scanomatic.generics.model as model


class FeaturesModel(model.Model):

    def __init__(self, analysis_directory="", email=""):

        self.analysis_directory = analysis_directory
        self.email = email

        super(FeaturesModel, self).__init__()