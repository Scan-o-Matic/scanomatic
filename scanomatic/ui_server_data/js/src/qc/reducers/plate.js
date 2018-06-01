// @flow
import type { Action } from '../actions';
import type { Plate as State } from '../state';

const initialState : State = { number: 0 };

export default function plate(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'PROJECT_SET':
        return initialState;
    case 'PLATE_SET':
        if (action.plate === state.number) return state;
        return { number: action.plate };
    case 'PLATE_GROWTHDATA_SET':
        if (action.plate !== state.number) return state;
        return Object.assign(
            {},
            state,
            { times: action.times, raw: action.raw, smooth: action.smooth },
        );
    case 'CURVE_FOCUS':
        if (action.plate !== state.number) return state;
        return Object.assign({}, state, { focus: { row: action.row, col: action.col } });
    default:
        return state;
    }
}
