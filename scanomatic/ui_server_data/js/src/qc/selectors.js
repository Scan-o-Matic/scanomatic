// @flow

import type {
    State,
    QualityIndexInfo,
    TimeSeries as _TimeSeries,
} from './state';

export type TimeSeries = _TimeSeries;

export function getProject(state: State): ?string {
    if (!state.settings) return null;
    return state.settings.project;
}

export function getPlate(state: State): ?number {
    if (!state.plate) return null;
    return state.plate.number;
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

export function getFocus(state: State) : ?QualityIndexInfo {
    if (!state.plate || !state.plate.qIndexQueue) return null;
    return state.plate.qIndexQueue[state.plate.qIndex];
}

export function getQIndexFromPosition(state: State, row: number, col: number) : ?number {
    if (!state.plate || !state.plate.qIndexQueue) return null;
    return state.plate.qIndexQueue
        .filter(item => item.row === row && item.col === col)
        .map(item => item.idx)[0];
}
