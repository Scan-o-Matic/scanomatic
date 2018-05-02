// @flow
import type { Action } from '../../actions';
import type { Scanners as State } from '../../state';

const initialState : State = new Map();

export default function scanners(state: State = initialState, action: Action): State {
    switch (action.type) {
    default:
        return state;
    }
}
