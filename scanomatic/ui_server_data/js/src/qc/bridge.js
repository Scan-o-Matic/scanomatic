// @flow

import {
    retrievePlateCurves, setProject, setPlate,
    setQualityIndexQueue, nextQualityIndex, previousQualityIndex, setQualityIndex,
} from './actions';
import {
    getRawCurve, getSmoothCurve, getTimes, getPlate,
    getFocus, getQIndexFromPosition,
} from './selectors';

import type { Action, ThunkAction } from './actions';
import type {
    State, TimeSeries, QualityIndexInfo, QualityIndexQueue,
} from './state';

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

    getPlate() : ?number {
        const state = this.store.getState();
        return getPlate(state);
    }

    getFocus() : ?QualityIndexInfo {
        const state = this.store.getState();
        return getFocus(state);
    }

    getQIndexFromPosition(row: number, col: number) : ?number {
        const state = this.store.getState();
        return getQIndexFromPosition(state, row, col);
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

    setQualityIndexQueue(queue: QualityIndexQueue, plate: ?number) {
        if (plate != null) this.setPlate(plate);
        this.store.dispatch(setQualityIndexQueue(queue));
    }

    setQualityIndex(index: number) {
        this.store.dispatch(setQualityIndex(index));
    }

    nextQualityIndex() {
        this.store.dispatch(nextQualityIndex());
    }

    previousQualityIndex() {
        this.store.dispatch(previousQualityIndex());
    }

    retrievePlateCurves(plate: ?number = null) {
        if (plate != null) {
            this.setPlate(plate);
        }
        this.store.dispatch(retrievePlateCurves());
    }
}

export default function Bridge(store: Store) :
{actions: Actions, selectors: Selectors, subscribe:(() => void) => void} {
    const actions = new Actions(store);
    const selectors = new Selectors(store);
    const subscribe: (() => void) => void = callback => store.subscribe(callback);
    return {
        actions,
        selectors,
        subscribe,
    };
}
