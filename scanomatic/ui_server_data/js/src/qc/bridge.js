// @flow

import { retrievePlateCurves, setProject } from './actions';
import { getCurve } from './selectors';
import type { State } from './state';

export class Selectors {
    constructor(store : State) {
        this.store = store;
    }

    getRawCurve(plate: Number, row : Number, col : Number) {
        return getCurve(this.store, row, col);
    }

    getSmoothCurve(plate: Number, row : Number, col : Number) {
        return getCurve(this.store, row, col);
    }
}

export class Actions {
    constructor(store : State) {
        this.store = store;
    }

    setProject(project: string) {
        this.store.dispatch(setProject(project));
    }

    retrievePlateCurves(plate: Number) {
        this.store.dispatch(retrievePlateCurves(plate));
    }
}

export default function Bridge(store: State) {
    const actions = new Actions(store);
    const selectors = new Selectors(store);
    return { actions, selectors };
}
