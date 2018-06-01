// @flow
import { getProject, getPlate } from './selectors';
import type { State, TimeSeries, PlateOfTimeSeries } from './state';
import { getPlateGrowthData } from './api';

export type Action
    = {| type: 'PLATE_SET', plate: number |}
    | {| type: 'PROJECT_SET', project: string |}
    | {| type: 'PINNING_SET', plate: number, rows: number, cols: number |}
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

export function setPinning(plate : number, rows: number, cols: number) : Action {
    return {
        type: 'PINNING_SET', plate, rows, cols,
    };
}

export function setPlateGrothData(
    plate: number,
    times: TimeSeries,
    smooth: PlateOfTimeSeries,
    raw: PlateOfTimeSeries,
) : Action {
    return {
        type: 'PLATE_GROWTHDATA_SET',
        plate,
        times,
        smooth,
        raw,
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
        getPlateGrowthData(project, plate).then((r) => {
            const { smooth, raw, times } = r;
            dispatch(setPlateGrothData(plate, times, smooth, raw));
            const rows = raw.length;
            const cols = raw[0].length;
            dispatch(setPinning(plate, rows, cols));
        });
    };
}
