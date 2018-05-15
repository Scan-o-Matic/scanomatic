import newExperiment from './newExperiment';

describe('projects/reducers/forms/newExperiment.js', () => {
    it('should return null as initial state', () => {
        expect(newExperiment(undefined, {})).toBe(null);
    });

    it('should return empty values on NEWEXPERIMENT_INIT', () => {
        const action = { type: 'NEWEXPERIMENT_INIT', projectId: 'foo' };
        expect(newExperiment(null, action)).toEqual({
            projectId: 'foo',
            submitted: false,
            fields: {
                name: '',
                description: '',
                duration: 0,
                interval: 0,
                scannerId: '',
                pinning: ['1536', '1536', '1536', '1536'],
            },
        });
    });

    const defaultState = {
        projectId: 'abc',
        submitted: false,
        fields: {
            name: 'A name',
            description: 'A description',
            duration: 42,
            interval: 4,
            scannerId: 'xyz',
        },
    };

    it('should update name on NEWEXPERIMENT_CHANGE', () => {
        const oldState = {
            ...defaultState,
            fields: { ...defaultState.fields, name: 'foo' },
        };
        const action = { type: 'NEWEXPERIMENT_CHANGE', field: 'name', value: 'bar' };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            fields: { ...defaultState.fields, name: 'bar' },
        });
    });

    it('should update description on NEWEXPERIMENT_CHANGE', () => {
        const oldState = {
            ...defaultState,
            fields: { ...defaultState.fields, description: 'foo' },
        };
        const action = { type: 'NEWEXPERIMENT_CHANGE', field: 'description', value: 'bar' };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            fields: { ...defaultState.fields, description: 'bar' },
        });
    });

    it('should update scannerId on NEWEXPERIMENT_CHANGE', () => {
        const oldState = {
            ...defaultState,
            fields: { ...defaultState.fields, scannerId: 'foo' },
        };
        const action = { type: 'NEWEXPERIMENT_CHANGE', field: 'scannerId', value: 'bar' };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            fields: { ...defaultState.fields, scannerId: 'bar' },
        });
    });

    it('should change duration on NEWEXPERIMENT_CHANGE', () => {
        const oldState = {
            ...defaultState,
            fields: { ...defaultState.fields, duration: 1 },
        };
        const action = { type: 'NEWEXPERIMENT_CHANGE', field: 'duration', value: 2 };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            fields: { ...defaultState.fields, duration: 2 },
        });
    });

    it('should change interval on NEWEXPERIMENT_CHANGE', () => {
        const oldState = {
            ...defaultState,
            fields: { ...defaultState.fields, interval: 1 },
        };
        const action = { type: 'NEWEXPERIMENT_CHANGE', field: 'interval', value: 2 };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            fields: { ...defaultState.fields, interval: 2 },
        });
    });

    it('should change pinning on NEWEXPERIMENT_CHANGE', () => {
        const oldState = {
            ...defaultState,
            fields: { ...defaultState.fields, pinning: ['384', '1536'] },
        };
        const action = { type: 'NEWEXPERIMENT_CHANGE', field: 'pinning', value: ['384', '6144'] };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            fields: { ...defaultState.fields, pinning: action.value },
        });
    });

    it('should set submitted to true on NEWEXPERIMENT_SUBMIT', () => {
        const oldState = { ...defaultState, submitted: false };
        const action = { type: 'NEWEXPERIMENT_SUBMIT' };
        expect(newExperiment(oldState, action)).toEqual({
            ...defaultState,
            submitted: true,
        });
    });

    it('should clear the state on NEWEXPERIMENT_CLEAR', () => {
        const action = { type: 'NEWEXPERIMENT_CLEAR' };
        expect(newExperiment(defaultState, action)).toBe(null);
    });
});
