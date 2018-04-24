// @flow
import type { NewExperiment as State } from '../../state';
import type { Action } from '../../actions';

export default function newExperiment(state: State = null, action: Action): State {
    switch (action.type) {
    case 'NEWEXPERIMENT_INIT':
        return {
            projectId: action.projectId,
            submitted: false,
            fields: {
                name: '',
                description: '',
                duration: 0,
                interval: 0,
                scannerId: '',
            },
        };
    case 'NEWEXPERIMENT_CHANGE':
        if (state == null) return state;
        switch (action.field) {
        case 'name':
        case 'description':
        case 'scannerId':
            return {
                ...state,
                fields: {
                    ...state.fields,
                    [action.field]: action.value,
                },
            };
        case 'duration':
        case 'interval':
            return {
                ...state,
                fields: {
                    ...state.fields,
                    [action.field]: action.value,
                },
            };
        default:
            return state;
        }
    case 'NEWEXPERIMENT_SUBMIT':
        if (state == null) return state;
        return {
            ...state,
            submitted: true,
        };
    case 'NEWEXPERIMENT_CLEAR':
        return null;
    default:
        return state;
    }
}
