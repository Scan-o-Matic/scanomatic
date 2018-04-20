// @flow
import type { NewProject as State } from '../../state';
import type { Action } from '../../actions';

const defaultState: State = {
    fields: {
        name: { value: '', touched: false },
        description: { value: '', touched: false },
    },
    error: null,
};

export default function reducer(state: State = defaultState, action: Action): State {
    switch (action.type) {
    case 'NEWPROJECT_CHANGE':
        if (action.field === 'name') {
            return {
                ...state,
                fields: {
                    ...state.fields,
                    name: {
                        value: action.value,
                        touched: true,
                    },
                },
            };
        }
        if (action.field === 'description') {
            return {
                ...state,
                fields: {
                    ...state.fields,
                    description: {
                        value: action.value,
                        touched: true,
                    },
                },
            };
        }
        return state;
    case 'NEWPROJECT_SUBMIT':
        return {
            ...state,
            fields: {
                ...state.fields,
                description: {
                    ...state.fields.description,
                    touched: true,
                },
                name: {
                    ...state.fields.name,
                    touched: true,
                },
            },
        };
    default:
        return state;
    }
}
