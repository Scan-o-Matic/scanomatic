// @flow
import type { NewProject as State } from '../../state';
import type { Action } from '../../actions';

const defaultState: State = {
    submitted: false,
    fields: {
        name: '',
        description: '',
    },
};

export default function reducer(state: State = null, action: Action): State {
    switch (action.type) {
    case 'NEWPROJECT_INIT':
        return defaultState;
    case 'NEWPROJECT_CHANGE':
        if (state == null) return null;
        return {
            ...state,
            fields: {
                ...state.fields,
                [action.field]: action.value,
            },
        };
    case 'NEWPROJECT_SUBMIT':
        if (state == null) return null;
        return { ...state, submitted: true };
    case 'NEWPROJECT_CLEAR':
        return null;
    default:
        return state;
    }
}
