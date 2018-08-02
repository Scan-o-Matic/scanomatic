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
    +phenotypes?: PlateValueArray,
    +badData?: PlateCoordinatesArray,
    +empty?: PlateCoordinatesArray,
    +noGrowth?: PlateCoordinatesArray,
    +undecidedProblem?: PlateCoordinatesArray,
};

export type State = {
    +settings: Settings,
    +plate: Plate,
};
