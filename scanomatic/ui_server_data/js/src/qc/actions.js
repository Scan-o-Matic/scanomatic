// @flow
import { getProject, getPlate, getPinning } from './selectors';
import type { State, TimeSeries } from './state';
import { getCurveData } from './api';

export type Action
    = {| type: 'PLATE_SET', plate: number |}
    | {| type: 'PROJECT_SET', project: string |}
    | {| type: 'CURVE_RAW_SET', plate: number, row: number, col: number, data: TimeSeries |}
    | {| type: 'CURVE_SMOOTH_SET', plate: number, row: number, col: number, data: TimeSeries |}
    | {| type: 'PINNING_SET', plate: number, rows: number, cols: number |}
    | {| type: 'TIMES_SET', times: TimeSeries, plate: number |}

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

export function setTimes(plate: number, times: TimeSeries) : Action {
    return {
        type: 'TIMES_SET', plate, times,
    };
}

export function setRawCurveData(
    plate: number,
    row: number,
    col: number,
    data: TimeSeries,
) : Action {
    return {
        type: 'CURVE_RAW_SET', plate, row, col, data,
    };
}

export function setSmoothCurveData(
    plate: number,
    row: number,
    col: number,
    data: TimeSeries,
) : Action {
    return {
        type: 'CURVE_SMOOTH_SET', plate, row, col, data,
    };
}

export type ThunkAction = (dispatch: Action => any, getState: () => State) => any;

// Limit on FF and Chrome (IE has more, but who cares?)
const MAX_CONCURRENT_CONNECTIONS = 6;
const POLL_INTERVAL = 50;

export function retrievePlateCurves() : ThunkAction {
    return (dispatch, getState) => {
        const state = getState();
        const project = getProject(state);
        if (project == null) {
            throw new Error('Cannot retrieve curves if project not set');
        }
        const plate = getPlate(state);
        const { rows, cols } = getPinning(state, plate) || { rows: 0, cols: 0 };
        let row = 0;
        let col = -1; // It will be increased to 0 on first poll
        let pending = 0;

        const success = (r) => {
            pending -= 1;
            dispatch(setRawCurveData(plate, r.row, r.col, r.raw));
            dispatch(setSmoothCurveData(plate, r.row, r.col, r.smooth));
            if (r.row === 0 && r.col === 0) {
                dispatch(setTimes(r.plate, r.times));
            }
        };
        const fail = () => {
            pending -= 1;
        };

        const poller = () => {
            if (getPlate(getState()) !== plate) return;
            while (pending < MAX_CONCURRENT_CONNECTIONS) {
                // Next position
                col += 1;
                if (col >= cols) {
                    row += 1;
                    col = 0;
                    if (row >= rows) return;
                }

                pending += 1;
                getCurveData(
                    project,
                    plate,
                    row,
                    col,
                )
                    .then(success)
                    .catch(fail);
            }
            setTimeout(poller, POLL_INTERVAL);
        };
        poller();
    };
}
