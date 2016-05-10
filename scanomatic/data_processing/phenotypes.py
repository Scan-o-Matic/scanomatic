from enum import Enum
from growth_phenotypes import Phenotypes
from curve_phase_phenotypes import VectorPhenotypes, CurvePhaseMetaPhenotypes


class PhenotypeDataType(Enum):

    Scalar = 0
    Vector = 1
    Phases = 2

    Trusted = 10
    UnderDevelopment = 11
    Other = 12

    All = 100

    def __call__(self, phenotype=None):

        _vectors = (VectorPhenotypes.PhasesClassifications,
                    VectorPhenotypes.PhasesPhenotypes,
                    Phenotypes.GrowthVelocityVector)

        _phases = (VectorPhenotypes.PhasesClassifications,
                   VectorPhenotypes.PhasesPhenotypes)

        _trusted = (Phenotypes.GenerationTime,
                    Phenotypes.ChapmanRichardsFit,
                    Phenotypes.ColonySize48h,
                    Phenotypes.InitialValue,
                    Phenotypes.ExperimentBaseLine,
                    Phenotypes.ExperimentGrowthYield,
                    Phenotypes.ExperimentPopulationDoublings,
                    Phenotypes.GenerationTimeWhen,
                    Phenotypes.ExperimentEndAverage,
                    Phenotypes.GenerationTimeStErrOfEstimate)

        _under_development = (Phenotypes.GenerationTimePopulationSize,
                              Phenotypes.GrowthLag,
                              CurvePhaseMetaPhenotypes.BimodalGrowthFirstImpulseDoubingTime,
                              CurvePhaseMetaPhenotypes.BimodalGrowthSecondImpulseDoubingTime,
                              CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution,
                              CurvePhaseMetaPhenotypes.InitialLag,
                              VectorPhenotypes.PhasesPhenotypes,
                              VectorPhenotypes.PhasesClassifications,
                              VectorPhenotypes)

        if self is PhenotypeDataType.Scalar:

            if phenotype is None:
                return tuple(p for p in Phenotypes if p not in _vectors)

            return phenotype not in _vectors

        elif self is PhenotypeDataType.Vector:

            if phenotype is None:
                return _vectors

            return phenotype in _vectors

        elif self is PhenotypeDataType.Phases:

            if phenotype is None:
                return _phases
            return phenotype in _phases

        elif self is PhenotypeDataType.Trusted:

            if phenotype is None:
                return _trusted

            return phenotype in _trusted

        elif self is PhenotypeDataType.UnderDevelopment:

            if phenotype is None:
                return _under_development

            return phenotype in _trusted or phenotype in _under_development

        elif self is PhenotypeDataType.Other:

            if phenotype is None:
                return tuple(p for p in Phenotypes if p not in _trusted and p not in _under_development)
            return phenotype not in _trusted and phenotype not in _under_development

        elif self is PhenotypeDataType.All:

            if phenotype is None:
                return tuple(p for p in Phenotypes)

            return True


def get_sort_order(phenotype):
    """

    :param phenotype: the phenotype name
    :return:
    """
    # TODO: Add the inverse exceptions
    return 1
