// @flow

import type {
    State,
    QualityIndexInfo,
    TimeSeries as _TimeSeries,
    PlateValueArray as _PlateValueArray,
    PlateCoordinatesArray as _PlateCoordinatesArray,
    Phenotype,
    QCMarks,
    QCMarkType,
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
    return state.plate.phenotypes[phenotype];
}

export function getCurrentPhenotypeData(state: State): ?PlateValueArray {
    const phenotype = getPhenotype(state);
    if (!state.plate || !state.plate.phenotypes || !phenotype) return null;
    return state.plate.phenotypes[phenotype];
}

export function getCurrentPhenotypeQCMarks(state: State): ?QCMarks {
    const phenotype = getPhenotype(state);
    if (!state.plate || !state.plate.qcmarks || !phenotype) return null;
    return state.plate.qcmarks[phenotype];
}

function isMarked(data: ?PlateCoordinatesArray, row: number, col: number) : bool {
    if (!data) return false;
    for (let i = 0; i < data[0].length; i += 1) {
        if (data[0][i] === row && data[1][i] === col) return true;
    }
    return false;
}

function getCurveQCMark(state: State, phenotype: Phenotype): ?QCMarkType {
    const focus = getFocus(state);
    if (!focus) return null;
    const { row, col } = focus;
    if (!state.plate || !state.plate.qcmarks) return null;
    const marks = state.plate.qcmarks[phenotype];
    if (isMarked(marks.badData, row, col)) return 'BadData';
    if (isMarked(marks.noGrowth, row, col)) return 'NoGrowth';
    if (isMarked(marks.empty, row, col)) return 'Empty';
    if (isMarked(marks.undecidedProblem, row, col)) return 'UndecidedProblem';
    return 'OK';
}

export function getFocusCurveQCMark(state: State): ?QCMarkType {
    const phenotype = getPhenotype(state);
    if (!phenotype) return null;
    return getCurveQCMark(state, phenotype);
}

export function getFocusCurveQCMarkAllPhenotypes(state: State): ?Object<Phenotype, QCMarkType> {
    if (!state.plate || !state.plate.qcmarks) return null;
    return Object.assign(
        {},
        ...Object.keys(state.plate.qcmarks)
            .map(phenotype => ({
                [phenotype]: getCurveQCMark(state, phenotype),
            })),
    );
}
