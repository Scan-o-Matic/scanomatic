// @flow
import type { State, Experiments, Scanners } from './state';
import { getScanners, getScanningJobs } from '../api';

export type Action
    = {| type: 'SCANNERS_SET', scanners: Scanners, date: Date |}
    | {| type: 'EXPERIMENTS_SET', experiments: Experiments, date: Date |}

export function setExperiments(experiments: Experiments) : Action {
    return { type: 'EXPERIMENTS_SET', experiments, date: new Date() };
}

export function setScanners(scanners : Scanners) : Action {
    return { type: 'SCANNERS_SET', scanners, date: new Date() };
}

type ThunkAction = (dispatch: Action => any, getState: () => State) => any;

export function retrieveStatus() : ThunkAction {
    return (dispatch) => {
        getScanners()
            .then(scanners => dispatch(setScanners(scanners.map(s => ({
                id: s.identifier,
                name: s.name,
                isOnline: s.power,
            })))));
        getScanningJobs()
            .then(jobs => dispatch(setExperiments(jobs.map(j => ({
                id: j.identifier,
                name: j.name,
                scannerId: j.scannerId,
                started: j.startTime && j.startTime.getTime(),
                end: j.startTime && j.duration.after(j.startTime).getTime(),
                stopped: j.terminationTime && j.terminationTime.getTime(),
            })))));
    };
}
