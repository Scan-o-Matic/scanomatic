// @flow
import type { Action } from '../actions';
import type { Experiments as State } from '../state';

const initialState : State = [];

export default function experiments(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'EXPERIMENTS_SET':
        return action.experiment.filter(e => e.started);
    default:
        return state;
    }
}
