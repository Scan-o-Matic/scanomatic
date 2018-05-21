// @flow
import type { State, Scanners, Experiments } from './state';

export function getExperiments(state: State): Experiments {
    return state.experiments;
}

export function getScanners(state: State): Scanners {
    return state.scanners;
}

export function hasLoadedScannersAndExperiments(state : State): boolean {
    const { scanners, experiments } = state.updateStatus;
    return scanners && experiments;
}
