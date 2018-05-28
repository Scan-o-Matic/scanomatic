import type { Action } from '../actions';
import type { Plate as State } from '../state';
import type { getPlate } from './selectors';

const initialState : State = { number: 1 };

export function GetUpdated2DArrayCopy(old, row, col, data) {
    const arr = [];
    const ly = Math.max(old ? old.length : 0, row + 1);
    for (let y = 0; y < ly; y += 1) {
        const copy = old ? old[y].slice() : [];
        if (y === row) {
            copy[col] = data;
        }
        arr.push(copy);
    }
    return arr;
}

export default function plate(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'PROJECT_SET':
        return initialState;
    case 'PLATE_SET':
        return Object.assign({}, state, { number: action.plate });
    case 'CURVE_RAW_SET': {
        if (action.plate !== getPlate()) return state;
        const raw = GetUpdated2DArrayCopy(state.raw, action.row, action.col, action.data);
        return Object.assign({}, state, { raw });
    }
    case 'CURVE_SMOOTH_SET': {
        if (action.plate !== getPlate()) return state;
        const smooth = GetUpdated2DArrayCopy(state.smooth, action.row, action.col, action.data);
        return Object.assign({}, state, { smooth });
    }
    default:
        return state;
    }
}
