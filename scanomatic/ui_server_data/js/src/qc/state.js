// @flow
export type TimeSeries = Array<number>;

export type PlateOfTimeSeries = Array<Array<TimeSeries>>;

export type QualityIndexInfo = {
    +idx: number,
    +col: number,
    +row: number,
};

export type QualityIndexQueue = Array<QualityIndexInfo>;

export type Settings = {
    +project?: string,
    +phenotype?: string,
};

export type Plate = {
    +number: number,
    +qIndex: number,
    +qIndexQueue?: QualityIndexQueue,
    +raw?: PlateOfTimeSeries,
    +smooth?: PlateOfTimeSeries,
    +times?: TimeSeries,
};

export type State = {
    +settings: Settings,
    +plate: Plate,
};
