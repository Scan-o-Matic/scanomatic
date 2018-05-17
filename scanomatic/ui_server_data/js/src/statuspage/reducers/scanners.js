// @flow
import type { Action } from '../actions';
import type { Scanners as State } from '../state';

const initialState : State = [];

export default function scanners(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'SCANNERS_SET':
        return action.scanners;
    default:
        return state;
    }
}
