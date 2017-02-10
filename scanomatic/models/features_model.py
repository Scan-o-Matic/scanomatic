from enum import Enum
import scanomatic.generics.model as model


class FeatureExtractionData(Enum):

    Default = 0
    State = 1


class FeaturesModel(model.Model):

    def __init__(self, analysis_directory="", email="", extraction_data=FeatureExtractionData.Default):

        self.analysis_directory = analysis_directory
        self.email = email
        self.extraction_data = extraction_data

        super(FeaturesModel, self).__init__()