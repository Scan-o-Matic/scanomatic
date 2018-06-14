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
}

export type PlatePosition = {
    +row: number,
    +col: number,
}

export type Plate = {
    +number: number,
    +qIndex: number,
    +qIndexQueue?: QualityIndexQueue,
    +raw?: PlateOfTimeSeries,
    +smooth?: PlateOfTimeSeries,
    +times?: TimeSeries,
    +focus?: PlatePosition,
}

export type State = {
    +settings: Settings,
    +plate: Plate,
};
