// @flow
import type { Action } from '../actions';
import type { UpdateStatus as State } from '../state';

const initialState : State = {
    experiments: false,
    scanners: false,
};

export default function updateStatus(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'SCANNERS_SET':
        return Object.assign({}, state, { scanners: true });
    case 'EXPERIMENTS_SET':
        return Object.assign({}, state, { experiments: true });
    default:
        return state;
    }
}
