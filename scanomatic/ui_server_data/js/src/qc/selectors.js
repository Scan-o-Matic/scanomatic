// @flow

import type { State, TimeSeries } from './state';

export function getPlate(state: State): Number {
    return state.plate.number;
}

export function getRawCurve(state: State, plate: Number, row: Number, col: Number): TimeSeries {
    const { raw, number: plateNumber } = state.plate;
    if (!raw || plate !== plateNumber) return null;
    return raw[row][col];
}

export function getSmoothCurve(state: State, plate: Number, row: Number, col: Number): TimeSeries {
    const { smooth, number: plateNumber } = state.plate;
    if (!smooth || plate !== plateNumber) return null;
    return smooth[row][col];
}
