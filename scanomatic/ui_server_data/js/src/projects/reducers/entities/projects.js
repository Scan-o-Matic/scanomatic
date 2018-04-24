// @flow
import type { Projects as State } from '../../state';
import type { Action } from '../../actions';

const defaultState: State = {};

export default function projects(state: State = defaultState, action: Action): State {
    switch (action.type) {
    case 'PROJECTS_ADD':
        return {
            ...state,
            [action.id]: {
                id: action.id,
                name: action.name,
                description: action.description,
                experimentIds: [],
            },
        };
    case 'EXPERIMENTS_ADD':
        return {
            ...state,
            [action.projectId]: {
                ...state[action.projectId],
                experimentIds: [action.id, ...state[action.projectId].experimentIds],
            },
        };
    default:
        return state;
    }
}
