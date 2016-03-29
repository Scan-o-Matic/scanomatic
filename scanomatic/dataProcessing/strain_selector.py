import numpy as np


def scan(plate_meta_data, column, value_function):

    return value_function(plate_meta_data[column])



class StrainSelector(object):

    def __init__(self, phenotyper, criteria):

        self.__phenotyper = phenotyper
        self.__criteria_string = criteria
        self.__criteria = StrainSelector.selection_criteria_from_string(phenotyper, criteria)

    def __filter(self, data):

        return tuple((d[f] if d is not None else None) for d, f in zip(data, self.__criteria))

    @staticmethod
    def parse_criteria_string(criteria_string):
        """
        Should take something like
        "ORF = YAWL031R | GENE = HOG1 | TOR1"
        look at column ORF after YAWL031R OR in column GENE after either HOG1 or TOR1

        If no number included instad of column name:
        "2 = HOG1" it implies look at second column

        Args:
            criteria_string:

        Returns:

        """

        # TODO: parser using re
        return tuple((column, value_function) for column, value_function in X)

    @staticmethod
    def selection_criteria_from_string(phenotyper, criteria_string):

        # TODO: Should act recursively maybe
        criteria = StrainSelector.parse_criteria_string(criteria_string)
        selection = [(scan(plate, column, value_function) for column, ) for plate in phenotyper.meta_data]

        #TODO: search using the parser
        return selection

    @property
    def raw_growth_data(self):

        return self.__filter(self.__phenotyper.raw_growth_data)

    @property
    def smooth_growth_data(self):

        return self.__filter(self.__phenotyper.smooth_growth_data)

    @property
    def phenotypes(self):

        # TODO: add list of active phenotypes

        phenotypes = []
        for phenotype in self.__phenotyper.included_phenotypes:

            phenotypes.append(self.get_phenotype(phenotype))

        # TODO: Re-arrange to be strain, phenotype
        phenotypes = np.array(phenotypes)
        return phenotypes

    @property
    def meta_data(self):

        # TODO: All indices from meta-data
        return ""

    def get_phenotype(self, phenotype):

        return self.__filter(self.__phenotyper.get_phenotype(phenotype))
