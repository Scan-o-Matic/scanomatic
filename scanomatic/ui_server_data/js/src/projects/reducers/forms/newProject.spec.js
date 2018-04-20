import reducer from './newProject';

fdescribe('projects/reducers/forms/newProject', () => {
    it('should return the default state', () => {
        expect(reducer(undefined, {})).toEqual({
            fields: {
                name: { value: '', touched: false },
                description: { value: '', touched: false },
            },
            error: null,
        });
    });

    const state = {
        fields: {
            name: { value: '', touched: false },
            description: { value: '', touched: false },
        },
        error: null,
    };

    it('should set the name on NEWPROJECT_CHANGE with name', () => {
        const action = { type: 'NEWPROJECT_CHANGE', field: 'name', value: 'foo' };
        expect(reducer(state, action).fields.name).toEqual({ value: 'foo', touched: true });
    });

    it('should set the description on NEWPROJECT_CHANGE with description', () => {
        const action = { type: 'NEWPROJECT_CHANGE', field: 'description', value: 'bar' };
        expect(reducer(state, action).fields.description).toEqual({ value: 'bar', touched: true });
    });

    it('should set touched to true on NEWPROJECT_SUBMIT', () => {
        const action = { type: 'NEWPROJECT_SUBMIT', field: 'name', value: 'foo' };
        expect(reducer(state, action)).toEqual({
            fields: {
                name: { value: '', touched: true },
                description: { value: '', touched: true },
            },
            error: null,
        });
    });
});

