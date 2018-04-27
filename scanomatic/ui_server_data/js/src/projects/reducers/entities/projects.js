// @flow
import type { Projects as State } from '../../state';
import type { Action } from '../../actions';

const defaultState: State = new Map();

export default function projects(state: State = defaultState, action: Action): State {
    switch (action.type) {
    case 'PROJECTS_ADD':
    {
        const newState = new Map(state);
        newState.set(action.id, {
            name: action.name,
            description: action.description,
            experimentIds: [],
        });
        return newState;
    }
    case 'EXPERIMENTS_ADD':
    {
        const project = state.get(action.projectId);
        if (!project) return state;
        const newState = new Map(state);
        newState.set(action.projectId, {
            ...project,
            experimentIds: [action.id, ...project.experimentIds],
        });
        return newState;
    }
    default:
        return state;
    }
}
