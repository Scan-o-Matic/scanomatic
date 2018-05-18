// @flow
import type { Action } from '../actions';
import type { UpdateStatus as State } from '../state';

const initialState : State = {
    experiments: null,
    scanners: null,
};

export default function updateStatus(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'SCANNERS_SET':
        return Object.assign({}, state, { scanners: action.date });
    case 'EXPERIMENTS_SET':
        return Object.assign({}, state, { experiments: action.date });
    default:
        return state;
    }
}
