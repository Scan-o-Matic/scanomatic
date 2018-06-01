// @flow

import type {
    State,
    TimeSeries as _TimeSeries,
    Pinning as _Pinning,
    PlatePosition as _PlatePosition,
} from './state';

export type TimeSeries = _TimeSeries;
export type Pinning = _Pinning;
export type PlatePosition = _PlatePosition;

export function getProject(state: State): ?string {
    if (!state.settings) return null;
    return state.settings.project;
}

export function getPlate(state: State): ?number {
    if (!state.plate) return null;
    return state.plate.number;
}

export function getPinning(state: State, plate: number): ?Pinning {
    if (!state.plate) return null;
    if (plate === state.plate.number) return state.plate.pinning;
    return null;
}

export function getTimes(state: State, plate: number): ?TimeSeries {
    if (!state.plate) return null;
    if (plate === state.plate.number) return state.plate.times;
    return null;
}

export function getRawCurve(state: State, plate: number, row: number, col: number): ?TimeSeries {
    if (!state.plate) return null;
    const { raw, number: plateNumber } = state.plate;
    if (!raw || plate !== plateNumber) return null;
    return raw[row][col];
}

export function getSmoothCurve(state: State, plate: number, row: number, col: number): ?TimeSeries {
    if (!state.plate) return null;
    const { smooth, number: plateNumber } = state.plate;
    if (!smooth || plate !== plateNumber) return null;
    return smooth[row][col];
}

export function getFocus(state: State) : ?PlatePosition {
    if (!state.plate || !state.plate.focus) return null;
    return state.plate.focus;
}
