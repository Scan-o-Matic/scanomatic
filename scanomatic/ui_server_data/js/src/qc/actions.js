// @flow
import type { State, TimeSeries } from './state';

export type Action
    = {| type: 'PLATE_SET', plate: Number |}
    | {| type: 'PROJECT_SET', project: string |}
    | {| type: 'CURVE_RAW_SET', plate: Number, row: Number, col: Number, data: TimeSeries |}
    | {| type: 'CURVE_SMOOTH_SET', plate: Number, row: Number, col: Number, data: TimeSeries |}

export function setPlate(plate : Number) : Action {
    return { type: 'PLATE_SET', plate };
}

export function setProject(project : string) : Action {
    return { type: 'PROJECT_SET', project };
}

export function setRawCurveData(
    plate: Number,
    row: Number,
    col: Number,
    data: TimeSeries,
) : Action {
    return {
        type: 'CURVE_RAW_SET', plate, row, col, data,
    };
}

export function setSmoothCurveData(
    plate: Number,
    row: Number,
    col: Number,
    data: TimeSeries,
) : Action {
    return {
        type: 'CURVE_SMOOTH_SET', plate, row, col, data,
    };
}

type ThunkAction = (dispatch: Action => any, getState: () => State) => any;

export function retrievePlateCurves() : ThunkAction {
    return () => {
        // dispatch arg if needed
        // getState second arg if needed
        // Perform fetching
        // Set plate curves
    };
}
