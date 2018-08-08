// @flow
import {
    getProject, getPlate, getPhenotype, getPhenotypeData as hasPhenotypeData,
    getFocusCurveQCMark, getFocus, getFocusCurveQCMarkAllPhenotypes, isDirty,
} from './selectors';
import type {
    State, TimeSeries, PlateOfTimeSeries, QualityIndexQueue,
    PlateValueArray, Phenotype, Mark, QCMarksMap,
} from './state';
import { getPlateGrowthData, getPhenotypeData, setCurveQCMark, setCurveQCMarkAll } from '../api';

export type Action
    = {| +type: 'PLATE_SET', plate: number |}
    | {| +type: 'PROJECT_SET', project: string |}
    | {| +type: 'CURVE_FOCUS', plate: number, row: number, col: number |}
    | {| +type: 'CURVE_QCMARK_SET', plate: number, row: number, col: number, mark: Mark, phenotype: ?Phenotype |}
    | {|
        +type: 'PLATE_GROWTHDATA_SET',
        plate: number,
        times: TimeSeries,
        smooth: PlateOfTimeSeries,
        raw: PlateOfTimeSeries,
    |}
    | {| +type: 'QUALITYINDEX_QUEUE_SET', queue: QualityIndexQueue |}
    | {| +type: 'QUALITYINDEX_SET', index: number |}
    | {| +type: 'QUALITYINDEX_NEXT' |}
    | {| +type: 'QUALITYINDEX_PREVIOUS' |}
    | {| +type: 'PHENOTYPE_SET', phenotype: Phenotype |}
    | {|
        +type: 'PLATE_PHENOTYPEDATA_SET',
        plate: number,
        phenotype: Phenotype,
        phenotypes: PlateValueArray,
        qcmarks: QCMarksMap,
    |}
    | {|
        +type: 'CURVE_QCMARK_SET',
        +plate: number,
        +row: number,
        +col: number,
        +phenotype: ?Phenotype,
        +mark: Mark,
        +dirty: bool,
    |}
    | {|
        +type: 'CURVE_QCMARK_REMOVEDIRTY',
        plate: number,
        row: number,
        col: number,
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
    qcmarks: QCMarksMap,
) : Action {
    return {
        type: 'PLATE_PHENOTYPEDATA_SET',
        plate,
        phenotype,
        phenotypes,
        qcmarks,
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
    mark: Mark,
    phenotype: ?Phenotype,
) : Action {
    return {
        type: 'CURVE_QCMARK_SET',
        phenotype,
        mark,
        plate,
        col,
        row,
        dirty: false,
    };
}

export function setStoreCurveQCMarkDirty(
    plate: number,
    row: number,
    col: number,
    mark: Mark,
    phenotype: ?Phenotype,
) : Action {
    return {
        type: 'CURVE_QCMARK_SET',
        phenotype,
        mark,
        plate,
        col,
        row,
        dirty: true,
    };
}

export function setQCMarkNotDirty(
    plate: number,
    row: number,
    col: number,
) : Action {
    return {
        type: 'CURVE_QCMARK_REMOVEDIRTY',
        plate,
        row,
        col,
    };
}


export type ThunkAction = (dispatch: Action => any, getState: () => State) => any;

export function updateFocusCurveQCMark(
    mark: Mark,
    phenotype: ?Phenotype,
    key: string,
) : ThunkAction {
    return (dispatch, getState) => {
        const state = getState();
        const project = getProject(state);
        if (!project) throw new Error('Cant set QC Mark if no project');
        const plate = getPlate(state);
        if (plate == null) throw new Error('Cant set QC Mark if no plate');
        const focus = getFocus(state);
        if (!focus) throw new Error('Cant set QC Mark if no focus');
        if (isDirty(state, plate, focus.row, focus.col)) {
            throw new Error('Cannot set mark while previous mark is still processing for this position');
        }

        dispatch(setStoreCurveQCMarkDirty(plate, focus.row, focus.col, mark, phenotype));

        let promise;
        if (phenotype) {
            promise = setCurveQCMark(project, plate, focus.row, focus.col, mark, phenotype, key);
        } else {
            promise = setCurveQCMarkAll(project, plate, focus.row, focus.col, mark, key);
        }

        return promise
            .then(() => {
                dispatch(setQCMarkNotDirty(plate, focus.row, focus.col));
                return Promise.resolve();
            })
            .catch(() => {
                // undo_preemtive curve_mark
                let previousMark;
                if (phenotype) {
                    previousMark = new Map([[phenotype, getFocusCurveQCMark(state)]]);
                } else {
                    previousMark = getFocusCurveQCMarkAllPhenotypes(state);
                }
                if (previousMark) {
                    previousMark
                        .forEach((prevMark, pheno) => dispatch(setStoreCurveQCMark(
                            plate,
                            focus.row,
                            focus.col,
                            prevMark || 'OK',
                            pheno,
                        )));
                }
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
            .map(phenotype => getPhenotypeData(project, plate, phenotype).then(data =>
                dispatch(setPlatePhenotypeData(plate, phenotype, data.phenotypes, data.qcmarks))));
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
                qcmarks,
                qIndexQueue,
            } = data;
            dispatch(setPlatePhenotypeData(plate, phenotype, phenotypes, qcmarks));
            dispatch(setQualityIndexQueue(qIndexQueue));
        });
    };
}
