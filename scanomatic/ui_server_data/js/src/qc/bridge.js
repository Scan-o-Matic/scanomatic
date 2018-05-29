// @flow

import { retrievePlateCurves, setProject, setPlate } from './actions';
import { getRawCurve, getSmoothCurve, getTimes, getPlate } from './selectors';
import type { State, TimeSeries } from './state';

export class Selectors {
    store: State;

    constructor(store : State) {
        this.store = store;
    }

    getRawCurve(plate: number, row : number, col : number) : TimeSeries {
        if (getPlate(this.store) !== plate) return null;
        return getRawCurve(this.store, row, col);
    }

    getSmoothCurve(plate: Number, row : Number, col : Number) : TimeSeries {
        if (getPlate(this.store) !== plate) return null;
        return getSmoothCurve(this.store, row, col);
    }

    getTimes(plate: number) : TimeSeries {
        if (getPlate(this.store) !== plate) return null;
        return getTimes(this.store);
    }
}

export class Actions {
    store: State;

    constructor(store : State) {
        this.store = store;
    }

    setProject(project: string) {
        this.store.dispatch(setProject(project));
    }

    setPlate(plate: number) {
        this.store.dispatch(setPlate(plate));
    }

    retrievePlateCurves(plate: Number) {
        this.store.dispatch(retrievePlateCurves(plate));
    }
}

export default function Bridge(store: State) {
    const actions = new Actions(store);
    const selectors = new Selectors(store);
    const subscribe: (() => void) => void = callback => this.store.subscribe(callback);
    return {
        actions,
        selectors,
        subscribe,
    };
}
