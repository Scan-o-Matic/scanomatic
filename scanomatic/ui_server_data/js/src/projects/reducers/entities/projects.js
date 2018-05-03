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
    case 'EXPERIMENTS_REMOVE':
    {
        let projectId;
        state.forEach((value, key) => {
            if (value.experimentIds.indexOf(action.id) > -1) {
                projectId = key;
            }
        });
        if (!projectId) return state;
        const project = state.get(projectId);
        const newState = new Map(state);
        newState.set(projectId, {
            ...project,
            experimentIds: project.experimentIds.filter(id => id !== action.id),
        });
        return newState;
    }
    default:
        return state;
    }
}
