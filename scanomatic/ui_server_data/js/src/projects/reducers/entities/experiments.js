// @flow
import type { Action } from '../../actions';
import type { Experiments as State } from '../../state';

const initialState : State = {};

export default function experiments(state: State = initialState, action: Action): State {
    switch (action.type) {
    case 'EXPERIMENTS_ADD':
        return {
            ...state,
            [action.id]: {
                name: action.name,
                description: action.description,
                duration: action.duration,
                interval: action.interval,
                started: null,
                stopped: null,
                reason: null,
                scanner: action.scanner,
            },
        };
    default:
        return state;
    }
}
