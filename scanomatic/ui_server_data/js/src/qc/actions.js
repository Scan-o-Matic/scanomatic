// @flow
import {
    getProject, getPlate, getPhenotype, getPhenotypeData as hasPhenotypeData,
    getFocusCurveQCMark, getFocus, getFocusCurveQCMarkAllPhenotypes,
} from './selectors';
import type {
    State, TimeSeries, PlateOfTimeSeries, QualityIndexQueue,
    PlateValueArray, PlateCoordinatesArray, Phenotype, QCMarkType,
} from './state';
import { getPlateGrowthData, getPhenotypeData, setCurveQCMark, setCurveQCMarkAll } from '../api';

export type Action
    = {| type: 'PLATE_SET', plate: number |}
    | {| type: 'PROJECT_SET', project: string |}
    | {| type: 'CURVE_FOCUS', plate: number, row: number, col: number |}
    | {| type: 'CURVE_QCMARK_SET', plate: number, row: number, col: number, mark: QCMarkType, phenotype: ?Phenotype |}
    | {|
        type: 'PLATE_GROWTHDATA_SET',
        plate: number,
        times: TimeSeries,
        smooth: PlateOfTimeSeries,
        raw: PlateOfTimeSeries,
    |}
    | {| type: 'QUALITYINDEX_QUEUE_SET', queue: QualityIndexQueue |}
    | {| type: 'QUALITYINDEX_SET', index: number |}
    | {| type: 'QUALITYINDEX_NEXT' |}
    | {| type: 'QUALITYINDEX_PREVIOUS' |}
    | {| type: 'PHENOTYPE_SET', phenotype: Phenotype |}
    | {|
        type: 'PLATE_PHENOTYPEDATA_SET',
        plate: number,
        phenotype: Phenotype,
        phenotypes: PlateValueArray,
    |}
    | {|
        type: 'PLATE_PHENOTYPEQC_SET',
        plate: number,
        phenotype: Phenotype,
        badData: PlateCoordinatesArray,
        empty: PlateCoordinatesArray,
        noGrowth: PlateCoordinatesArray,
        undecidedProblem: PlateCoordinatesArray
    |}
    | {|
        type: 'CURVE_QCMARK_SET',
        phenotype: ?Phenotype,
        mark: QCMarkType,
    |}

export function setPlate(plate : number) : Action {
    return { type: 'PLATE_SET', plate };
}

export function setProject(project : string) : Action {
    return { type: 'PROJECT_SET', project };
}

export function setPhenotype(phenotype: Phenotype) : Action {
    return { type: 'PHENOTYPE_SET', phenotype };
}

export function setPlateGrowthData(
    plate: number,
    times: TimeSeries,
    raw: PlateOfTimeSeries,
    smooth: PlateOfTimeSeries,
) : Action {
    return {
        type: 'PLATE_GROWTHDATA_SET',
        plate,
        times,
        raw,
        smooth,
    };
}

export function setPlatePhenotypeData(
    plate: number,
    phenotype: Phenotype,
    phenotypes: PlateValueArray,
) : Action {
    return {
        type: 'PLATE_PHENOTYPEDATA_SET',
        plate,
        phenotype,
        phenotypes,
    };
}

export function setPhenotypeQCMarks(
    plate: number,
    phenotype: Phenotype,
    badData: PlateCoordinatesArray,
    empty: PlateCoordinatesArray,
    noGrowth: PlateCoordinatesArray,
    undecidedProblem: PlateCoordinatesArray,
) : Action {
    return {
        type: 'PLATE_PHENOTYPEQC_SET',
        plate,
        phenotype,
        badData,
        empty,
        noGrowth,
        undecidedProblem,
    };
}

export function focusCurve(
    plate: number,
    row: number,
    col: number,
) : Action {
    return {
        type: 'CURVE_FOCUS', plate, row, col,
    };
}

export function setQualityIndexQueue(queue: QualityIndexQueue) : Action {
    return {
        type: 'QUALITYINDEX_QUEUE_SET',
        queue,
    };
}

export function setQualityIndex(index: number): Action {
    return {
        type: 'QUALITYINDEX_SET',
        index,
    };
}

export function nextQualityIndex() : Action {
    return { type: 'QUALITYINDEX_NEXT' };
}

export function previousQualityIndex() : Action {
    return { type: 'QUALITYINDEX_PREVIOUS' };
}

export function setStoreCurveQCMark(
    plate: number,
    row: number,
    col: number,
    mark: QCMarkType,
    phenotype: ?Phenotype,
) : Action {
    return {
        type: 'CURVE_QCMARK_SET',
        phenotype,
        mark,
        plate,
        col,
        row,
    };
}

export type ThunkAction = (dispatch: Action => any, getState: () => State) => any;

export function updateFocusCurveQCMark(
    mark: QCMarkType,
    phenotype: ?Phenotype,
    key: string,
) : ThunkAction {
    return (dispatch, getState) => {
        let promise;
        const state = getState();
        const project = getProject(state);
        if (!project) throw new Error('Cant set QC Mark if no project');
        const plate = getPlate(state);
        if (!plate) throw new Error('Cant set QC Mark if no plate');
        const focus = getFocus(state);
        if (!focus) throw new Error('Cant set QC Mark if no focus');
        let previousMark;
        if (phenotype) {
            previousMark = { [phenotype]: getFocusCurveQCMark(state) };
            promise = setCurveQCMark(project, plate, focus.row, focus.col, mark, phenotype, key);
        } else {
            previousMark = getFocusCurveQCMarkAllPhenotypes(state);
            promise = setCurveQCMarkAll(project, plate, focus.row, focus.col, mark, key);
        }
        return promise.catch(() => {
            // undo_preemtive curve_mark
            Object.entries(previousMark)
                .forEach(([pheno, prevMark]) => dispatch(setStoreCurveQCMark(
                    plate,
                    focus.row,
                    focus.col,
                    prevMark,
                    pheno,
                )));
            return Promise.resolve();
        });
    };
}

export function retrievePlateCurves() : ThunkAction {
    return (dispatch, getState) => {
        const state = getState();
        const project = getProject(state);
        if (project == null) {
            throw new Error('Cannot retrieve curves if project not set');
        }
        const plate = getPlate(state);
        // getPlate can only return null when project is not set, so
        // this can't really happen, but linter gets upset for type
        // mismatch for the api call below if this check is not performed.
        if (plate == null) {
            throw new Error('Cannot retrieve curves if project not set');
        }

        return getPlateGrowthData(project, plate).then((r) => {
            const { smooth, raw, times } = r;
            dispatch(setPlateGrowthData(plate, times, raw, smooth));
        });
    };
}

export function retrievePhenotypesNeededInGraph(plate: number) : ThunkAction {
    return (dispatch, getState) => {
        const state = getState();
        const project = getProject(state);
        if (project == null) {
            throw new Error('Cannot retrieve phenotype if project not set');
        }
        const currentPlate = getPlate(state);
        if (plate !== currentPlate) return Promise.resolve();

        const promises = ['GenerationTime', 'GenerationTimeWhen', 'ExperimentGrowthYield']
            .filter(phenotype => !hasPhenotypeData(state, phenotype))
            .map(phenotype => getPhenotypeData(project, plate, phenotype).then((data) => {
                dispatch(setPlatePhenotypeData(plate, phenotype, data.phenotypes));
                dispatch(setPhenotypeQCMarks(
                    plate,
                    phenotype,
                    data.badData,
                    data.empty,
                    data.noGrowth,
                    data.undecidedProblem,
                ));
            }));
        return Promise.all(promises);
    };
}

export function retrievePlatePhenotype(plate: number) : ThunkAction {
    return (dispatch, getState) => {
        const state = getState();
        const project = getProject(state);
        if (project == null) {
            throw new Error('Cannot retrieve phenotype if project not set');
        }
        const phenotype = getPhenotype(state);
        if (phenotype == null) {
            throw new Error('Cannot retrieve phenotype if phenotype not set');
        }
        const currentPlate = getPlate(state);
        if (currentPlate === plate && hasPhenotypeData(state, phenotype)) return Promise.resolve();
        if (currentPlate !== plate) dispatch(setPlate(plate));
        return getPhenotypeData(project, plate, phenotype).then((data) => {
            const {
                phenotypes,
                badData,
                empty,
                noGrowth,
                undecidedProblem,
                qIndexQueue,
            } = data;
            dispatch(setPlatePhenotypeData(plate, phenotype, phenotypes));
            dispatch(setPhenotypeQCMarks(
                plate,
                phenotype,
                badData,
                empty,
                noGrowth,
                undecidedProblem,
            ));
            dispatch(setQualityIndexQueue(qIndexQueue));
        });
    };
}
