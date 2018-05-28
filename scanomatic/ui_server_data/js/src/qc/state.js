// @flow
export type TimeSeries = Array<Number>;

export type Settings = {
    +project: string,
}

export type Plate = {
    +number: Number,
    +raw: Array<Array<TimeSeries>>,
    +smooth: Array<Array<TimeSeries>>,
}

export type State = {
    +settings: Settings,
    +plate: Plate,
};
