import numpy as np


def scan(plate_meta_data, column, value_function):

    return value_function(plate_meta_data[column])


class StrainSelector(object):

    def __init__(self, phenotyper, selection):

        self.__phenotyper = phenotyper
        self.__selection = selection

    def __add__(self, other):

        if other.__phenotyper == self.__phenotyper:
            return StrainSelector(self.__phenotyper,
                                  tuple(StrainSelector.__joined(s1, s2) for s1, s2 in zip(self.__selection,
                                                                                          other.__selection)))
        else:
            raise ValueError("Other does not have matching phenotyper")

    def __iadd__(self, other):

        if other.__phenotyper == self.__phenotyper:
            self.__selection = tuple(StrainSelector.__joined(s1, s2) for s1, s2 in zip(self.__selection,
                                                                                       other.__selection))
        else:
            raise ValueError("Other does not have matching phenotyper")

    @staticmethod
    def __joined(selection1, selection2):

        if selection1 and selection2:
            return selection1[0] + selection2[0], selection1[1] + selection2[1]
        elif selection1:
            return selection1
        else:
            return selection2

    def __filter(self, data):

        return tuple((d[f] if d is not None else None) for d, f in zip(data, self.__selection))

    @property
    def selection(self):

        return self.__selection

    @property
    def raw_growth_data(self):

        return self.__filter(self.__phenotyper.raw_growth_data)

    @property
    def smooth_growth_data(self):

        return self.__filter(self.__phenotyper.smooth_growth_data)

    @property
    def phenytype_names(self):

        return [phenotype.name for phenotype in self.__phenotyper.analysed_phenotypes]

    @property
    def phenotypes(self):

        phenotypes = {}
        for phenotype in self.__phenotyper.analysed_phenotypes:

            phenotypes[phenotype] = self.get_phenotype(phenotype)

        return np.array(
            tuple(
                tuple(phenotypes[p][i] for p in self.__phenotyper.analysed_phenotypes)
                for i, _ in enumerate(self.__selection)))

    @property
    def vector_phenotypes(self):

        # TODO: something here
        pass

    @property
    def meta_data(self):

        md = self.__phenotyper.meta_data
        return [tuple(md.get_data_from_numpy_where(i, s) if s else None) for i, s in enumerate(self.__selection)]

    def get_phenotype(self, phenotype):

        return self.__filter(self.__phenotyper.get_phenotype(phenotype))
