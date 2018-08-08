// @flow
import type { Action } from '../actions';
import type { Settings as State } from '../state';

const initialState : State = { showNormalized: false };

export default function settings(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'PROJECT_SET':
        return { project: action.project };
    case 'PHENOTYPE_SET':
        return Object.assign({}, state, { phenotype: action.phenotype });
    case 'SHOWNORMALIZED_SET':
        return Object.assign({}, state, { showNormalized: action.value });
    default:
        return state;
    }
}
