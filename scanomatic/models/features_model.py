__author__ = 'martin'

import scanomatic.generics.model as model


class FeaturesModel(model.Model):

    def __init__(self, analysis_directory=""):

        self.analysis_directory = analysis_directory

        super(FeaturesModel, self).__init__()