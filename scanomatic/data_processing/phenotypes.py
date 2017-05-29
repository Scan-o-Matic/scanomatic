from enum import Enum
from itertools import chain

from growth_phenotypes import Phenotypes
from scanomatic.data_processing.phases.features import VectorPhenotypes, CurvePhaseMetaPhenotypes


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

        _vectors = (
            VectorPhenotypes.PhasesClassifications,
            VectorPhenotypes.PhasesPhenotypes,
            Phenotypes.GrowthVelocityVector,
        )

        _phases = (
            VectorPhenotypes.PhasesClassifications,
            VectorPhenotypes.PhasesPhenotypes,
        )

        _trusted = (
            Phenotypes.GenerationTime,
            Phenotypes.ChapmanRichardsFit,
            Phenotypes.ChapmanRichardsParam1,
            Phenotypes.ChapmanRichardsParam2,
            Phenotypes.ChapmanRichardsParam3,
            Phenotypes.ChapmanRichardsParam4,
            Phenotypes.ChapmanRichardsParamXtra,
            Phenotypes.ColonySize48h,
            Phenotypes.InitialValue,
            Phenotypes.ExperimentBaseLine,
            Phenotypes.ExperimentGrowthYield,
            Phenotypes.ExperimentPopulationDoublings,
            Phenotypes.GenerationTimeWhen,
            Phenotypes.ExperimentEndAverage,
            Phenotypes.GenerationTimeStErrOfEstimate,
            VectorPhenotypes.PhasesPhenotypes,
            VectorPhenotypes.PhasesClassifications,
        )

        _under_development = (
            Phenotypes.GenerationTimePopulationSize,
            Phenotypes.GrowthLag,
            Phenotypes.Monotonicity,
            Phenotypes.ExperimentLowPoint,
            Phenotypes.ExperimentLowPointWhen,
            Phenotypes.ResidualGrowth,
            Phenotypes.ResidualGrowthAsPopulationDoublings,
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
            CurvePhaseMetaPhenotypes.MajorImpulseFlankAsymmetry,
            CurvePhaseMetaPhenotypes.TimeBeforeMajorGrowth,
        )

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
                return set(chain(_under_development, _trusted))

            return phenotype in _trusted or phenotype in set(chain(_under_development, _trusted))

        elif self is PhenotypeDataType.Other:

            if phenotype is None:
                return tuple(p for p in Phenotypes if p not in _trusted and p not in _under_development)
            return phenotype not in _trusted and phenotype not in _under_development

        elif self is PhenotypeDataType.All:

            if phenotype is None:
                return tuple(p for p in chain(Phenotypes, VectorPhenotypes, CurvePhaseMetaPhenotypes))

            for pheno_type in (Phenotypes, VectorPhenotypes, CurvePhaseMetaPhenotypes):

                try:
                    if phenotype in pheno_type or pheno_type[phenotype]:
                        return True
                except KeyError:
                    pass

    @classmethod
    def classify(cls, phenotype):

        return tuple(t for t in cls if t(phenotype))


def infer_phenotype_from_name(name):

    for phenotype_class in (Phenotypes, CurvePhaseMetaPhenotypes, VectorPhenotypes):

        try:
            return phenotype_class[name]
        except KeyError:
            pass

    raise ValueError("Supplied name '{0}' not a known phenotype".format(name))


def get_sort_order(phenotype):
    """

    :param phenotype: the phenotype name
    :return:
    """
    # TODO: Add more inverse exceptions
    if phenotype in (Phenotypes.ExperimentGrowthYield,):
        return -1
    return 1
