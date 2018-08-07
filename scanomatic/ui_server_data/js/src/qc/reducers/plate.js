// @flow
import type { Action } from '../actions';
import type { Plate as State, QCMarkType, QCMarks, PlateCoordinatesArray } from '../state';

const initialState : State = { number: 0, qIndex: 0 };

function addMark(previous: ?PlateCoordinatesArray, row: number, col: number)
: PlateCoordinatesArray {
    if (!previous) return [[row], [col]];
    const next = [[], []];
    let found = false;
    for (let i = 0; i < previous[0].length; i += 1) {
        if (previous[0][i] === row && previous[1][i] === col) found = true;
        next[0].push(previous[0][i]);
        next[1].push(previous[1][i]);
    }
    if (!found) {
        next[0].push(row);
        next[1].push(col);
    }
    return next;
}

function removeMark(previous: ?PlateCoordinatesArray, row: number, col: number)
: ?PlateCoordinatesArray {
    if (!previous) return null;
    const next = [[], []];
    for (let i = 0; i < previous[0].length; i += 1) {
        if (previous[0][i] !== row && previous[1][i] !== col) {
            next[0].push(previous[0][i]);
            next[1].push(previous[1][i]);
        }
    }
    return next;
}

function updateQCMarks(marks: QCMarks, row: number, col: number, mark: QCMarkType) : QCMarks {
    return {
        noGrowth: mark === 'NoGrowth' ? addMark(marks.noGrowth, row, col) : removeMark(marks.noGrowth, row, col),
        empty: mark === 'Empty' ? addMark(marks.empty, row, col) : removeMark(marks.empty, row, col),
        badData: mark === 'BadData' ? addMark(marks.badData, row, col) : removeMark(marks.badData, row, col),
        undecidedProblem: mark === 'UndecidedProblem' ? addMark(marks.undecidedProblem, row, col) : removeMark(marks.undecidedProblem, row, col),
    };
}

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
    case 'PLATE_PHENOTYPEDATA_SET': {
        if (action.plate !== state.number) return state;
        const nextPhenotypes = new Map(state.phenotypes);
        nextPhenotypes.set(action.phenotype, action.phenotypes);
        const nextQC = new Map(state.qcmarks);
        nextQC.set(action.phenotype, action.qcmarks);
        return Object.assign(
            {},
            state,
            {
                phenotypes: nextPhenotypes,
                qcmarks: nextQC,
            },
        );
    }
    case 'CURVE_QCMARK_SET':
        if (action.plate !== state.number) return state;
        if (action.phenotype) {
            return Object.assign(
                {},
                state,
                {
                    qcmarks: Object.assign({}, state.qcmarks, {
                        [action.phenotype]: updateQCMarks(
                            state.qcmarks ? state.qcmarks[action.phenotype] : {},
                            action.row,
                            action.col,
                            action.mark,
                        ),
                    }),
                },
            );
        }
        return Object.assign(
            {},
            state,
            {
                qcmarks: Object.assign({}, ...Object.entries(state.qcmarks || {})
                    .map(([phenotype, marks]) => ({
                        [phenotype]: updateQCMarks(
                            marks,
                            action.row,
                            action.col,
                            action.mark,
                        ),
                    }))),
            },
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
