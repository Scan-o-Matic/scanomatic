// @flow
import type { Action } from '../actions';
import type { Plate as State } from '../state';

const initialState : State = { number: 0, qIndex: 0 };

export default function plate(state: State = initialState, action: Action) {
    switch (action.type) {
    case 'PROJECT_SET':
        return initialState;
    case 'PLATE_SET':
        if (action.plate === state.number) return state;
        return { number: action.plate, qIndex: 0 };
    case 'PLATE_GROWTHDATA_SET':
        if (action.plate !== state.number) return state;
        return Object.assign(
            {},
            state,
            { times: action.times, raw: action.raw, smooth: action.smooth },
        );
    case 'QUALITYINDEX_QUEUE_SET':
        return Object.assign(
            {},
            state,
            {
                qIndexQueue: action.queue
                    .sort((a, b) => a.idx - b.idx)
                    .map((item, idx) => ({ idx, col: item.col, row: item.row })),
            },
        );
    case 'QUALITYINDEX_SET':
        if (!state.qIndexQueue) return state;
        return Object.assign(
            {},
            state,
            { qIndex: Math.max(Math.min(action.index, state.qIndexQueue.length - 1), 0) },
        );
    case 'QUALITYINDEX_NEXT':
        if (!state.qIndexQueue) return state;
        return Object.assign(
            {},
            state,
            { qIndex: (state.qIndex + 1) % state.qIndexQueue.length },
        );
    case 'QUALITYINDEX_PREVIOUS': {
        if (!state.qIndexQueue) return state;
        let next = state.qIndex - 1;
        if (next < 0) next += state.qIndexQueue.length;
        return Object.assign({}, state, { qIndex: next });
    }
    default:
        return state;
    }
}
