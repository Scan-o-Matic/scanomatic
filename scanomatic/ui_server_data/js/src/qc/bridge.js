// @flow

import {
    retrievePlateCurves, setProject, setPlate, setPinning,
} from './actions';
import {
    getRawCurve, getSmoothCurve, getTimes, getPlate,
} from './selectors';

import type { Action, ThunkAction } from './actions';
import type { State, TimeSeries, Pinning } from './state';

type Store = {
    +dispatch: (Action | ThunkAction) => any,
    +getState: () => State,
    +subscribe: (() => any) => any,
}

class Selectors {
    store : Store

    constructor(store : Store) {
        this.store = store;
    }

    getRawCurve(plate: number, row : number, col : number) : ?TimeSeries {
        const state = this.store.getState();
        return getRawCurve(state, plate, row, col);
    }

    getSmoothCurve(plate: number, row : number, col : number) : ?TimeSeries {
        const state = this.store.getState();
        return getSmoothCurve(state, plate, row, col);
    }

    getTimes(plate: number) : ?TimeSeries {
        const state = this.store.getState();
        return getTimes(state, plate);
    }

    getPlate() : number {
        const state = this.store.getState();
        return getPlate(state);
    }
}

class Actions {
    store: Store;

    constructor(store : Store) {
        this.store = store;
    }

    setProject(project: string) {
        this.store.dispatch(setProject(project));
    }

    setPlate(plate: number) {
        this.store.dispatch(setPlate(plate));
    }

    setPinning(plate: number, rows: number, cols: number) {
        this.store.dispatch(setPinning(plate, rows, cols));
    }

    retrievePlateCurves(plate: ?number = null, pinning: ?Pinning = null) {
        if (plate != null) {
            this.setPlate(plate);
            if (pinning != null) this.setPinning(plate, pinning.rows, pinning.cols);
        }
        this.store.dispatch(retrievePlateCurves());
    }
}

export default function Bridge(store: Store) {
    const actions = new Actions(store);
    const selectors = new Selectors(store);
    const subscribe: (() => void) => void = callback => store.subscribe(callback);
    return {
        actions,
        selectors,
        subscribe,
    };
}
