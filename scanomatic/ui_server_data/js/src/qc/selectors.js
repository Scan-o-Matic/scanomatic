// @flow

import type {
    State,
    QualityIndexInfo,
    TimeSeries as _TimeSeries,
    PlateValueArray as _PlateValueArray,
    PlateCoordinatesArray as _PlateCoordinatesArray,
    Phenotype,
    Mark,
    QCMarksMap,
} from './state';

export type TimeSeries = _TimeSeries;
export type PlateValueArray = _PlateValueArray;
export type PlateCoordinatesArray = _PlateCoordinatesArray;

export function getProject(state: State): ?string {
    if (!state.settings) return null;
    return state.settings.project;
}

export function getPhenotype(state: State): ?Phenotype {
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

export function getPhenotypeData(state: State, phenotype: Phenotype): ?PlateValueArray {
    if (!state.plate || !state.plate.phenotypes) return null;
    return state.plate.phenotypes.get(phenotype);
}

export function getCurrentPhenotypeData(state: State): ?PlateValueArray {
    const phenotype = getPhenotype(state);
    if (!state.plate || !state.plate.phenotypes || !phenotype) return null;
    return state.plate.phenotypes.get(phenotype);
}

export function getCurrentPhenotypeQCMarks(state: State): ?QCMarksMap {
    const phenotype = getPhenotype(state);
    if (!state.plate || !state.plate.qcmarks || !phenotype) return null;
    return state.plate.qcmarks.get(phenotype);
}

function isMarked(data: ?PlateCoordinatesArray, row: number, col: number) : bool {
    if (!data) return false;
    for (let i = 0; i < data[0].length; i += 1) {
        if (data[0][i] === row && data[1][i] === col) return true;
    }
    return false;
}

function parseFocusCurveQCMark(state: State, phenotype: Phenotype): ?Mark {
    const focus = getFocus(state);
    if (!focus) return null;
    const { row, col } = focus;
    if (!state.plate || !state.plate.qcmarks) return null;
    const marks = state.plate.qcmarks.get(phenotype);
    if (!marks) return 'OK';
    if (isMarked(marks.get('BadData'), row, col)) return 'BadData';
    if (isMarked(marks.get('NoGrowth'), row, col)) return 'NoGrowth';
    if (isMarked(marks.get('Empty'), row, col)) return 'Empty';
    if (isMarked(marks.get('UndecidedProblem'), row, col)) return 'UndecidedProblem';
    return 'OK';
}

export function getFocusCurveQCMark(state: State): ?Mark {
    const phenotype = getPhenotype(state);
    if (!phenotype) return null;
    return parseFocusCurveQCMark(state, phenotype);
}

export function getFocusCurveQCMarkAllPhenotypes(state: State): ?Map<Phenotype, Mark> {
    if (!state.plate || !state.plate.qcmarks) return null;
    const marks = new Map();
    (state.plate.qcmarks || new Map())
        .forEach((_, phenotype) => marks.set(phenotype, parseFocusCurveQCMark(state, phenotype) || 'OK'));
    return marks;
}

export function isDirty(state: State, plate: number, row: number, col: number) : bool {
    if (!state.plate || !state.plate.dirty || getPlate(state) !== plate) return false;
    return state.plate.dirty
        .some(([dirtyRow, dirtyCol]) => dirtyRow === row && dirtyCol === col);
}
