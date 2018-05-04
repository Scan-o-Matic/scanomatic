import reducer from './projects';

describe('projects/reducers/entities/projects', () => {
    it('should return the default state', () => {
        expect(reducer(undefined, {})).toEqual(new Map());
    });

    it('should add a project on PROJECTS_ADD', () => {
        const state = new Map([
            ['p1', {
                name: 'Existing Project',
                description: '',
                experimentIds: ['e12'],
            }],
        ]);
        const action = {
            type: 'PROJECTS_ADD', id: 'p2', name: 'New Project', description: 'Very new',
        };
        expect(reducer(state, action)).toEqual(new Map([
            ['p1', {
                name: 'Existing Project',
                description: '',
                experimentIds: ['e12'],
            }],
            ['p2', {
                name: 'New Project',
                description: 'Very new',
                experimentIds: [],
            }],
        ]));
    });

    it('should handle EXPERIMENTS_ADD', () => {
        const state = new Map([
            ['p1', {
                name: 'Project',
                description: '',
                experimentIds: ['e12'],
            }],
            ['p2', {
                name: 'Other Project',
                description: '',
                experimentIds: ['e34'],
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
            projectId: 'p1',
        };
        expect(reducer(state, action)).toEqual(new Map([
            ['p1', {
                name: 'Project',
                description: '',
                experimentIds: ['003', 'e12'],
            }],
            ['p2', {
                name: 'Other Project',
                description: '',
                experimentIds: ['e34'],
            }],
        ]));
    });

    it('should handle EXPERIMENTS_REMOVE', () => {
        const state = new Map([
            ['p1', {
                name: 'Project',
                description: '',
                experimentIds: ['e12'],
            }],
            ['p2', {
                name: 'Other Project',
                description: '',
                experimentIds: ['e34'],
            }],
        ]);
        const action = {
            type: 'EXPERIMENTS_REMOVE',
            id: 'e34',
            date: new Date(),
        };
        expect(reducer(state, action)).toEqual(new Map([
            ['p1', {
                name: 'Project',
                description: '',
                experimentIds: ['e12'],
            }],
            ['p2', {
                name: 'Other Project',
                description: '',
                experimentIds: [],
            }],
        ]));
    });
});
