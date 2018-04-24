import reducer from './newProject';

describe('projects/reducers/forms/newProject', () => {
    it('should return null as initial state', () => {
        expect(reducer(undefined, {})).toBe(null);
    });

    it('should initialize the state on NEWPROJECT_INIT', () => {
        const action = { type: 'NEWPROJECT_INIT' };
        expect(reducer(null, action)).toEqual({
            submitted: false,
            fields: {
                name: '',
                description: '',
            },
        });
    });

    const state = {
        submitted: false,
        fields: {
            name: '',
            description: '',
        },
    };

    it('should set the name on NEWPROJECT_CHANGE with name', () => {
        const action = { type: 'NEWPROJECT_CHANGE', field: 'name', value: 'foo' };
        expect(reducer(state, action).fields.name).toEqual('foo');
    });

    it('should set the description on NEWPROJECT_CHANGE with description', () => {
        const action = { type: 'NEWPROJECT_CHANGE', field: 'description', value: 'bar' };
        expect(reducer(state, action).fields.description).toEqual('bar');
    });

    it('should set submitted to true on NEWPROJECT_SUBMIT', () => {
        const action = { type: 'NEWPROJECT_SUBMIT', field: 'name', value: 'foo' };
        expect(reducer(state, action)).toEqual({
            ...state,
            submitted: true,
        });
    });

    it('should clear the state on NEWPROJECT_CLEAR', () => {
        const action = { type: 'NEWPROJECT_CLEAR' };
        expect(reducer(state, action)).toBe(null);
    });
});

