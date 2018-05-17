// @flow
import type { Experiments, Scanners } from './state';

export type Action
    = {| type: 'SCANNERS_SET', scanners: Scanners, date: Date |}
    | {| type: 'EXPERIMENTS_SET', experiments: Experiments, date: Date |}

export function setExperiments(experiments: Experiments) : Action {
    return { type: 'EXPERIMENTS_SET', experiments, date: new Date() };
}

export function setScanners(scanners : Scanners) : Action {
    return { type: 'SCANNERS_SET', scanners, date: new Date() };
}
