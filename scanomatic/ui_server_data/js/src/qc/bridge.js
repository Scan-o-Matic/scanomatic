// @flow

import {
    retrievePlateCurves, setProject, setPlate, setPhenotype,
    setQualityIndexQueue, nextQualityIndex, previousQualityIndex, setQualityIndex,
} from './actions';
import {
    getRawCurve, getSmoothCurve, getTimes, getPlate, getProject,
    getFocus, getQIndexFromPosition, getPhenotype,
} from './selectors';

import type { Action, ThunkAction } from './actions';
import type {
    State, TimeSeries, QualityIndexInfo, QualityIndexQueue, Phenotype,
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

    getProject() : ?string {
        const state = this.store.getState();
        return getProject(state);
    }

    getPhenotype() : ?Phenotype {
        const state = this.store.getState();
        return getPhenotype(state);
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

    setPhenotype(phenotype: Phenotype) {
        this.store.dispatch(setPhenotype(phenotype));
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

export default class Bridge {
    actions: Actions;
    selectors: Selectors;
    store: Store;

    constructor(store: Store) {
        this.store = store;
        this.actions = new Actions(store);
        this.selectors = new Selectors(store);
    }

    subscribe(callback: () => void) {
        this.store.subscribe(callback);
    }
}
