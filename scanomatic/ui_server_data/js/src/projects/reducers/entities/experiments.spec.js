import experiments from './experiments';

describe('projects/reducers/entities/experiments', () => {
    it('should return an empty initial state', () => {
        expect(experiments(undefined, {})).toEqual({});
    });

    it('should handle EXPERIMENT_ADD', () => {
        const state = {
            '002': {
                name: 'Some experiment',
                description: 'This is an experiment',
                duration: 1200,
                interval: 300,
                started: null,
                stopped: null,
                reason: null,
                scanner: '001',
            },
        };
        const action = {
            type: 'EXPERIMENTS_ADD',
            description: '',
            duration: 357,
            id: '003',
            interval: 24,
            name: 'Other experiment',
            scanner: 'S04',
        };
        expect(experiments(state, action)).toEqual({
            ...state,
            '003': {
                name: 'Other experiment',
                description: '',
                duration: 357,
                interval: 24,
                started: null,
                stopped: null,
                reason: null,
                scanner: 'S04',
            },
        });
    });
});
