// @flow

import type {
    State,
    QualityIndexInfo,
    TimeSeries as _TimeSeries,
    PlateValueArray as _PlateValueArray,
    PlateCoordinatesArray as _PlateCoordinatesArray,
} from './state';

export type TimeSeries = _TimeSeries;
export type PlateValueArray = _PlateValueArray;
export type PlateCoordinatesArray = _PlateCoordinatesArray;
export type QCMarks =  {
    badData: PlateCoordinatesArray,
    noGrowth: PlateCoordinatesArray,
    empty: PlateCoordinatesArray,
    undecidedProblem: PlateCoordinatesArray,
};

export function getProject(state: State): ?string {
    if (!state.settings) return null;
    return state.settings.project;
}

export function getPhenotype(state: State): ?string {
    if (!state.settings) return null;
    return state.settings.phenotype;
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

export function getPhenotypeData(state: State): ?PlateValueArray {
    if (!state.plate) return null;
    return state.plate.phenotypes;
}

export function getPhenotypeQCMarks(state: State): ?QCMarks {
    if (!state.plate) return null;
    return {
        badData: state.plate.badData,
        empty: state.plate.empty,
        noGrowth: state.plate.noGrowth,
        undecidedProblem: state.plate.undecidedProblem,
    };
}
