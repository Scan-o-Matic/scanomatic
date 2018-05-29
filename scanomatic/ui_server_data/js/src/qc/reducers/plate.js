import type { Action } from '../actions';
import type { Plate as State } from '../state';
import { getUpdated2DArrayCopy } from '../helpers';

const initialState : State = { number: 0 };

export default function plate(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'PROJECT_SET':
        return initialState;
    case 'PLATE_SET':
        return Object.assign({}, state, { number: action.plate });
    case 'CURVE_RAW_SET': {
        if (action.plate !== state.number) return state;
        const raw = getUpdated2DArrayCopy(state.raw, action.row, action.col, action.data);
        return Object.assign({}, state, { raw });
    }
    case 'CURVE_SMOOTH_SET': {
        if (action.plate !== state.number) return state;
        const smooth = getUpdated2DArrayCopy(state.smooth, action.row, action.col, action.data);
        return Object.assign({}, state, { smooth });
    }
    case 'PINNING_SET':
        if (action.plate !== state.number) return state;
        return Object.assign({}, state, { pinning: { rows: action.rows, cols: action.cols } });
    case 'TIMES_SET':
        if (action.plate !== state.number) return state;
        return Object.assign({}, state, { times: action.times });
    default:
        return state;
    }
}
