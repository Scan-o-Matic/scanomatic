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

export type PhenotypeDataMap = Map<Phenotype, PlateValueArray>;

export type Mark = 'OK' | 'NoGrowth' | 'BadData' | 'Empty' | 'UndecidedProblem';
export type QCMarksMap = Map<Mark, PlateCoordinatesArray>;

export type PhenotypeQCMarksMap = Map<Phenotype, QCMarksMap>;

export type Plate = {
    +number: number,
    +qIndex: number,
    +qIndexQueue?: QualityIndexQueue,
    +raw?: PlateOfTimeSeries,
    +smooth?: PlateOfTimeSeries,
    +times?: TimeSeries,
    +phenotypes?: PhenotypeDataMap,
    +qcmarks?: PhenotypeQCMarksMap,
    +dirty?: Array<Array<number>>,
};

export type State = {
    +settings: Settings,
    +plate: Plate,
};
