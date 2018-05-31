// @flow

import type { State, TimeSeries, Pinning } from './state';

export function getProject(state: State): ?string {
    return state.settings.project;
}

export function getPlate(state: State): number {
    return state.plate.number;
}

export function hasStartedLoadingPlate(state: State): boolean {
    const { pinning, raw } = state.plate;
    if (!pinning || !raw) return false;
    return raw.length > 0;
}

export function getPinning(state: State, plate: number): ?Pinning {
    if (plate === state.plate.number) return state.plate.pinning;
    return null;
}

export function getTimes(state: State, plate: number): ?TimeSeries {
    if (plate === state.plate.number) return state.plate.times;
    return null;
}

export function getRawCurve(state: State, plate: number, row: number, col: number): ?TimeSeries {
    const { raw, number: plateNumber } = state.plate;
    if (!raw || plate !== plateNumber) return null;
    return raw[row][col];
}

export function getSmoothCurve(state: State, plate: number, row: number, col: number): ?TimeSeries {
    const { smooth, number: plateNumber } = state.plate;
    if (!smooth || plate !== plateNumber) return null;
    return smooth[row][col];
}
