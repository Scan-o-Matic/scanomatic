// @flow

import type { State, TimeSeries, Pinning } from './state';

export function getProject(state: State): string {
    return state.settings.project;
}

export function getPlate(state: State): number {
    return state.plate.number;
}

export function getPinning(state: State): Pinning {
    return state.plate.pinning;
}

export function getTimes(state: State): TimeSeries {
    return state.plate.times;
}

export function getRawCurve(state: State, plate: number, row: number, col: number): TimeSeries {
    const { raw, number: plateNumber } = state.plate;
    if (!raw || plate !== plateNumber) return null;
    return raw[row][col];
}

export function getSmoothCurve(state: State, plate: number, row: number, col: number): TimeSeries {
    const { smooth, number: plateNumber } = state.plate;
    if (!smooth || plate !== plateNumber) return null;
    return smooth[row][col];
}
