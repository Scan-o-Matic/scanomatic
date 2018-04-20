import { validateNewExperiment, validateNewProject } from './validators';

fdescribe('projects/validators', () => {
    describe('validateNewProject', () => {
        const validState = {
            fields: {
                name: 'Foobar',
                description: 'Foo bar baz',
            },
        };

        it('should return no errors given a valid state', () => {
            const errors = validateNewProject(validState);
            expect(errors.size).toEqual(0);
        });

        it('should return an error if name is empty', () => {
            const state = {
                ...validState,
                fields: {
                    ...validState.fields,
                    name: '',
                },
            };
            const errors = validateNewProject(state);
            expect(errors.get('name')).toEqual('Project name cannot be empty');
        });

        it('should not return an error if description is empty', () => {
            const state = {
                ...validState,
                fields: {
                    ...validState.fields,
                    description: '',
                },
            };
            const errors = validateNewProject(state);
            expect(errors.get('description')).not.toBeDefined();
        });
    });

    describe('validateNewExperiment', () => {
        const validState = {
            fields: {
                description: 'Foo bar baz',
                duration: 3000,
                interval: 300,
                name: 'Foobar',
                projectId: 'abc',
                scannerId: 'abc',
            },
        };

        it('should return no errors given a valid state', () => {
            const errors = validateNewExperiment(validState);
            expect(errors.size).toEqual(0);
        });

        it('should return an error if name is empty', () => {
            const state = {
                ...validState,
                fields: {
                    ...validState.fields,
                    name: '',
                },
            };
            const errors = validateNewExperiment(state);
            expect(errors.get('name')).toEqual('Required');
        });

        it('should return an error if scannerId is empty', () => {
            const state = {
                ...validState,
                fields: {
                    ...validState.fields,
                    scannerId: '',
                },
            };
            const errors = validateNewExperiment(state);
            expect(errors.get('scannerId')).toEqual('Required');
        });

        it('should return an error if duration is 0', () => {
            const state = {
                ...validState,
                fields: {
                    ...validState.fields,
                    duration: 0,
                },
            };
            const errors = validateNewExperiment(state);
            expect(errors.get('duration')).toEqual('Required');
        });

        it('should return an error if interval is 0', () => {
            const state = {
                ...validState,
                fields: {
                    ...validState.fields,
                    interval: 0,
                },
            };
            const errors = validateNewExperiment(state);
            expect(errors.get('interval')).toEqual('Required');
        });
    });
});
