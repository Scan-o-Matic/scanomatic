import experiments from './experiments';

describe('projects/reducers/entities/experiments', () => {
    it('should return an empty initial state', () => {
        expect(experiments(undefined, {})).toEqual(new Map());
    });

    it('should handle EXPERIMENT_ADD', () => {
        const state = new Map([
            ['002', {
                name: 'Some experiment',
                description: 'This is an experiment',
                duration: 1200,
                interval: 300,
                started: null,
                stopped: null,
                reason: null,
                scannerId: '001',
            }],
        ]);
        const action = {
            type: 'EXPERIMENTS_ADD',
            description: '',
            duration: 357,
            id: '003',
            interval: 24,
            name: 'Other experiment',
            scannerId: 'S04',
        };
        expect(experiments(state, action)).toEqual(new Map([
            ...state,
            ['003', {
                name: 'Other experiment',
                description: '',
                duration: 357,
                interval: 24,
                started: null,
                stopped: null,
                reason: null,
                scannerId: 'S04',
            }],
        ]));
    });
});
