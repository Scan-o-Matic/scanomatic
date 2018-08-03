// @flow
export type TimeSeries = Array<number>;

export type PlateOfTimeSeries = Array<Array<TimeSeries>>;

export type QualityIndexInfo = {
    +idx: number,
    +col: number,
    +row: number,
};

export type QualityIndexQueue = Array<QualityIndexInfo>;

export type PlateValueArray = Array<Array<number>>;
export type PlateCoordinatesArray = Array<Array<number>>; // [[y1, y2, ...], [x1, x2, ...]]

export type Phenotype = "GenerationTime"
    | "ExperimentGrowthYield"
    | "ChapmanRichardsParam1"
    | "ChapmanRichardsParam2"
    | "ChapmanRichardsParam3"
    | "ChapmanRichardsParam4"
    | "ChapmanRichardsParamXtra"
    | "ColonySize48h"
    | "ExperimentBaseLine"
    | "ExperimentPopulationDoublings"
    | "GenerationTimeStErrOfEstimate"
    | "GenerationTimeWhen"
    | "InitialValue";

export type Settings = {
    +project?: string,
    +phenotype?: Phenotype,
};

export type PhenotypeDataCollection = {
    +GenerationTime?: PlateValueArray,
    +ExperimentGrowthYield?: PlateValueArray,
    +ChapmanRichardsParam1?: PlateValueArray,
    +ChapmanRichardsParam2?: PlateValueArray,
    +ChapmanRichardsParam3?: PlateValueArray,
    +ChapmanRichardsParam4?: PlateValueArray,
    +ChapmanRichardsParamXtra?: PlateValueArray,
    +ColonySize48h?: PlateValueArray,
    +ExperimentBaseLine?: PlateValueArray,
    +ExperimentPopulationDoublings?: PlateValueArray,
    +GenerationTimeStErrOfEstimate?: PlateValueArray,
    +GenerationTimeWhen?: PlateValueArray,
    +InitialValue?: PlateValueArray,
};

export type QCMarks = {
    +badData?: PlateCoordinatesArray,
    +empty?: PlateCoordinatesArray,
    +noGrowth?: PlateCoordinatesArray,
    +undecidedProblem?: PlateCoordinatesArray,
};

export type QCMarksCollection = {
    +GenerationTime?: QCMarks,
    +ExperimentGrowthYield?: QCMarks,
    +ChapmanRichardsParam1?: QCMarks,
    +ChapmanRichardsParam2?: QCMarks,
    +ChapmanRichardsParam3?: QCMarks,
    +ChapmanRichardsParam4?: QCMarks,
    +ChapmanRichardsParamXtra?: QCMarks,
    +ColonySize48h?: QCMarks,
    +ExperimentBaseLine?: QCMarks,
    +ExperimentPopulationDoublings?: QCMarks,
    +GenerationTimeStErrOfEstimate?: QCMarks,
    +GenerationTimeWhen?: QCMarks,
    +InitialValue?: QCMarks,
};

export type Plate = {
    +number: number,
    +qIndex: number,
    +qIndexQueue?: QualityIndexQueue,
    +raw?: PlateOfTimeSeries,
    +smooth?: PlateOfTimeSeries,
    +times?: TimeSeries,
    +phenotypes?: PhenotypeDataCollection,
    +qcmarks?: QCMarksCollection,
};

export type State = {
    +settings: Settings,
    +plate: Plate,
};
