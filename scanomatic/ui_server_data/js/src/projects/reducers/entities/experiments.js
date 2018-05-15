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
            done: null,
            end: null,
        });
        return newState;
    }
    case 'EXPERIMENTS_START':
    {
        const newState = new Map(state);
        const experiment = state.get(action.id);
        newState.set(
            action.id,
            Object.assign(
                {},
                experiment,
                { started: action.date },
            ),
        );
        return newState;
    }
    case 'EXPERIMENTS_REMOVE':
    {
        const newState = new Map(state);
        newState.delete(action.id);
        return newState;
    }
    case 'EXPERIMENTS_STOP':
    {
        const { reason, date, id } = action;
        const newState = new Map(state);
        const experiment = state.get(action.id);
        let storeReason = null;
        if (reason != null && reason.length > 0) storeReason = reason;
        newState.set(
            id,
            Object.assign(
                {},
                experiment,
                {
                    stopped: date,
                    reason: storeReason,
                },
            ),
        );
        return newState;
    }
    case 'EXPERIMENTS_DONE':
    {
        const newState = new Map(state);
        const experiment = state.get(action.id);
        newState.set(
            action.id,
            Object.assign(
                {},
                experiment,
                { done: true },
            ),
        );
        return newState;
    }
    case 'EXPERIMENTS_REOPEN':
    {
        const newState = new Map(state);
        const experiment = state.get(action.id);
        newState.set(
            action.id,
            Object.assign(
                {},
                experiment,
                { done: false },
            ),
        );
        return newState;
    }
    default:
        return state;
    }
}
