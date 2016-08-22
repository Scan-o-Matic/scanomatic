from enum import Enum
from growth_phenotypes import Phenotypes
from curve_phase_phenotypes import VectorPhenotypes, CurvePhaseMetaPhenotypes


class PhenotypeDataType(Enum):
    """The enum contains two types of phenotype classifications.

    There are three that deal with data-source/type `Scalar`, `Vector`, `Phases`.
    There are three that deal with stage `Trusted`, `UnderDevelopment`, `Other`.

    _NOTE_: A stage will always include the more trusted stages phenotypes too.
    So to see what phenotypes are actually under development one needs to do:
    ```set(PhenotypeDataType.UnderDevelopment).difference(PhenotypeDataTypes.Trusted)```

    To test if a phenotype is of a certain type you do:
     ```set(PhenotypeDataType.UnderDevelopment()).difference(PhenotypeDataType.Trusted())```.

    Attributes:
        PhenotypeDataType.Scalar: The phenotype is scalar, this is the default expectation
        PhenotypeDataType.Vector: These are the phenotype that are entire vectors:
            [`VectorPhenotypes.PhasesPhenotypes`, `VectorPhenotypes.PhasesClassification`,
             `Phenotypes.GrowthVelocityVector`]
        PhenotypeDataType.Phases: The two vector phenotypes above that clearly deals with phases.
        PhenotypeDataType.Trusted: Phenotypes that have been verified and are unlikely to change.
        PhenotypeDataType.UnderDevelopment: Phenotypes that are very likely to change and may include
            bugs and errors.
        PhenotypeDataType.Other: Typically disused or discarded phenotypes.
        PhenotypeDataType.All: All growth phenotypes.

    Methods:
        classify: List the types that a phenotype fulfills.

    """
    Scalar = 0
    """:type : PhenotypeDataType"""
    Vector = 1
    """:type : PhenotypeDataType"""
    Phases = 2
    """:type : PhenotypeDataType"""

    Trusted = 10
    """:type : PhenotypeDataType"""
    UnderDevelopment = 11
    """:type : PhenotypeDataType"""
    Other = 12
    """:type : PhenotypeDataType"""

    All = 100
    """:type : PhenotypeDataType"""

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
                              Phenotypes.ExperimentLowPoint,
                              Phenotypes.ExperimentLowPointWhen,
                              CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution,
                              CurvePhaseMetaPhenotypes.MajorImpulseAveragePopulationDoublingTime,
                              CurvePhaseMetaPhenotypes.FirstMinorImpulseYieldContribution,
                              CurvePhaseMetaPhenotypes.FirstMinorImpulseAveragePopulationDoublingTime,
                              CurvePhaseMetaPhenotypes.InitialLag,
                              CurvePhaseMetaPhenotypes.InitialLagAlternativeModel,
                              CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteAngle,
                              CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteAngle,
                              CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteIntersect,
                              CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteIntersect,
                              CurvePhaseMetaPhenotypes.Modalities,
                              CurvePhaseMetaPhenotypes.ModalitiesAlternativeModel,
                              CurvePhaseMetaPhenotypes.Collapses,
                              CurvePhaseMetaPhenotypes.ExperimentDoublings,
                              CurvePhaseMetaPhenotypes.ResidualGrowth,
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

    @classmethod
    def classify(cls, phenotype):

        return tuple(t for t in cls if t(phenotype))


def get_sort_order(phenotype):
    """

    :param phenotype: the phenotype name
    :return:
    """
    # TODO: Add the inverse exceptions
    return 1
