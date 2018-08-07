// @flow
import { getProject, getPlate, getPhenotype, getPhenotypeData as hasPhenotypeData } from './selectors';
import type {
    State, TimeSeries, PlateOfTimeSeries, QualityIndexQueue,
    PlateValueArray, Phenotype, QCMarksMap,
} from './state';
import { getPlateGrowthData, getPhenotypeData } from '../api';

export type Action
    = {| type: 'PLATE_SET', plate: number |}
    | {| type: 'PROJECT_SET', project: string |}
    | {| type: 'CURVE_FOCUS', plate: number, row: number, col: number |}
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
        qcmarks: QCMarksMap,
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

export type ThunkAction = (dispatch: Action => any, getState: () => State) => any;

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
