// @flow
import { getProject, getPlate } from './selectors';
import type { State, TimeSeries, PlateOfTimeSeries } from './state';
import { getPlateGrowthData } from '../api';

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

export function setPlate(plate : number) : Action {
    return { type: 'PLATE_SET', plate };
}

export function setProject(project : string) : Action {
    return { type: 'PROJECT_SET', project };
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

export function focusCurve(
    plate: number,
    row: number,
    col: number,
) : Action {
    return {
        type: 'CURVE_FOCUS', plate, row, col,
    };
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
