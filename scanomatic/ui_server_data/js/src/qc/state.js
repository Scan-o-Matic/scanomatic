// @flow
export type TimeSeries = Array<number>;

export type PlateOfTimeSeries = Array<Array<TimeSeries>>;

export type Settings = {
    +project: string,
}

export type Plate = {
    +number: Number,
    +raw: PlateOfTimeSeries,
    +smooth: PlateOfTimeSeries,
}

export type State = {
    +settings: Settings,
    +plate: Plate,
};
