// @flow
import type { Action } from '../actions';
import type { Plate as State, Mark, QCMarksMap, PlateCoordinatesArray } from '../state';

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
: PlateCoordinatesArray {
    if (!previous) return [[], []];
    const next = [[], []];
    for (let i = 0; i < previous[0].length; i += 1) {
        if (previous[0][i] !== row && previous[1][i] !== col) {
            next[0].push(previous[0][i]);
            next[1].push(previous[1][i]);
        }
    }
    return next;
}

function updateQCMarks(marks: QCMarksMap, row: number, col: number, mark: Mark) : QCMarksMap {
    return new Map(['NoGrowth', 'Empty', 'BadData', 'UndecidedProblem']
        .map(markType => [
            markType,
            mark === markType ?
                addMark(marks.get(markType), row, col) :
                removeMark(marks.get(markType), row, col),
        ]));
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
    case 'CURVE_QCMARK_SET': {
        if (action.plate !== state.number) return state;
        const nextQC = new Map(state.qcmarks);

        if (action.phenotype) {
            nextQC.set(
                action.phenotype,
                updateQCMarks(
                    nextQC.get(action.phenotype) || new Map(),
                    action.row,
                    action.col,
                    action.mark,
                ),
            );
        } else {
            nextQC.forEach((marks, phenotype) => nextQC.set(phenotype, updateQCMarks(
                marks || new Map(),
                action.row,
                action.col,
                action.mark,
            )));
        }
        return Object.assign({}, state, {
            qcmarks: nextQC,
            dirty: action.dirty ?
                (state.dirty || []).concat([[action.row, action.col]]) :
                (state.dirty || [])
                    .filter(([row, col]) => action.row !== row || action.col !== col),
        });
    }
    case 'CURVE_QCMARK_REMOVEDIRTY':
        if (action.plate !== state.number) return state;
        return Object.assign({}, state, {
            dirty: (state.dirty || [])
                .filter(([row, col]) => action.row !== row || action.col !== col),
        });
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
