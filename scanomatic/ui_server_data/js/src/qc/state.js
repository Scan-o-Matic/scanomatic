// @flow
export type TimeSeries = Array<number>;

export type PlateOfTimeSeries = Array<Array<TimeSeries>>;

export type Settings = {
    +project?: string,
}

export type Pinning = {
    +rows: number,
    +cols: number,
}

export type PlatePosition = {
    +row: number,
    +col: number,
}

export type Plate = {
    +number: number,
    +raw?: PlateOfTimeSeries,
    +smooth?: PlateOfTimeSeries,
    +pinning?: Pinning,
    +times?: TimeSeries,
    +focus?: PlatePosition,
}

export type State = {
    +settings: Settings,
    +plate: Plate,
};
