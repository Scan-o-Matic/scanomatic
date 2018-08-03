// @flow
import { getProject, getPlate, getPhenotype, getPhenotypeData as hasPhenotypeData } from './selectors';
import type {
    State, TimeSeries, PlateOfTimeSeries, QualityIndexQueue,
    PlateValueArray, PlateCoordinatesArray, Phenotype,
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

export function retrieveGraphPhenotypes(plate: number) : ThunkAction {
    return (dispatch, getState) => {
        const state = getState();
        const project = getProject(state);
        if (project == null) {
            throw new Error('Cannot retrieve phenotype if project not set');
        }
        const currentPlate = getPlate(state);
        if (plate == null || currentPlate !== plate) {
            throw new Error('Cannot retrieve phenotype if plate not set');
        }
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
        if (currentPlate === plate) return Promise.resolve();
        if (plate == null) {
            throw new Error('Cannot retrieve phenotype if plate not set');
        }
        dispatch(setPlate(plate));
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
            retrieveGraphPhenotypes(plate);
        });
    };
}
