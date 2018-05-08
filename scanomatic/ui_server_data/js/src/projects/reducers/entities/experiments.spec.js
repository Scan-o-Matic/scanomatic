import experiments from './experiments';

describe('projects/reducers/entities/experiments', () => {
    const state = new Map([
        ['002', {
            name: 'Some experiment',
            description: 'This is an experiment',
            duration: 1200,
            interval: 300,
            started: null,
            stopped: null,
            reason: null,
            end: null,
            done: null,
            scannerId: '001',
        }],
    ]);

    it('should return an empty initial state', () => {
        expect(experiments(undefined, {})).toEqual(new Map());
    });

    it('should handle EXPERIMENT_ADD', () => {
        const action = {
            type: 'EXPERIMENTS_ADD',
            description: '',
            duration: 357,
            id: '003',
            interval: 24,
            name: 'Other experiment',
            scannerId: 'S04',
            pinning: new Map([[1: '6144']]),
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
                end: null,
                done: null,
                scannerId: 'S04',
                pinning: new Map([[1: '6144']]),
            }],
        ]));
    });

    it('should handle EXPERIMENTS_START', () => {
        const date = new Date();
        const action = {
            type: 'EXPERIMENTS_START',
            id: '002',
            date,
        };
        expect(experiments(state, action)).toEqual(new Map([
            ['002', {
                name: 'Some experiment',
                description: 'This is an experiment',
                duration: 1200,
                interval: 300,
                started: date,
                stopped: null,
                reason: null,
                end: null,
                done: null,
                scannerId: '001',
            }],
        ]));
    });

    it('should handle EXPERIMENTS_DONE', () => {
        const action = {
            type: 'EXPERIMENTS_DONE',
            id: '002',
        };
        expect(experiments(state, action)).toEqual(new Map([
            ['002', {
                name: 'Some experiment',
                description: 'This is an experiment',
                duration: 1200,
                interval: 300,
                started: null,
                stopped: null,
                reason: null,
                end: null,
                done: true,
                scannerId: '001',
            }],
        ]));
    });

    it('should handle EXPERIMENTS_REOPEN', () => {
        const action = {
            type: 'EXPERIMENTS_REOPEN',
            id: '002',
        };
        expect(experiments(state, action)).toEqual(new Map([
            ['002', {
                name: 'Some experiment',
                description: 'This is an experiment',
                duration: 1200,
                interval: 300,
                started: null,
                stopped: null,
                reason: null,
                end: null,
                done: false,
                scannerId: '001',
            }],
        ]));
    });

    it('should handle EXPERIMENTS_REMOVE', () => {
        const action = {
            type: 'EXPERIMENTS_REMOVE',
            id: '002',
        };
        expect(experiments(state, action)).toEqual(new Map([]));
    });

});
