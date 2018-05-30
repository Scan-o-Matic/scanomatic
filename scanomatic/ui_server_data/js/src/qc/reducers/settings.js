// @flow
import type { Action } from '../actions';
import type { Settings as State } from '../state';

const initialState : State = {};

export default function settings(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'PROJECT_SET':
        return { project: action.project };
    default:
        return state;
    }
}
