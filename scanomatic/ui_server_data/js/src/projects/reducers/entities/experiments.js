// @flow
import type { Action } from '../../actions';
import type { Experiments as State } from '../../state';

const initialState : State = new Map();

export default function experiments(state: State = initialState, action: Action): State {
    switch (action.type) {
    case 'EXPERIMENTS_ADD':
    {
        const newState = new Map(state);
        newState.set(action.id, {
            name: action.name,
            description: action.description,
            duration: action.duration,
            interval: action.interval,
            started: null,
            stopped: null,
            reason: null,
            scannerId: action.scannerId,
        });
        return newState;
    }
    default:
        return state;
    }
}
