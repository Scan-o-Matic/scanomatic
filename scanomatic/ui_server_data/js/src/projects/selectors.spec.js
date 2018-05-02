import * as selectors from '../../src/projects/selectors';
import StateBuilder from './StateBuilder';

describe('projects/selectors', () => {
    describe('getProjects', () => {
        it('should return an empty array if there are no projects', () => {
            const state = new StateBuilder().clearProjects().build();
            expect(selectors.getProjects(state)).toEqual([]);
        });
        it('should return the projects reverse-ordered by id', () => {
            const state = new StateBuilder()
                .clearProjects()
                .addProject({ id: 'P1', name: 'Bar', description: 'I am also a project' })
                .addProject({ id: 'P2', name: 'Foo', description: 'I am a project' })
                .build();
            expect(selectors.getProjects(state)).toEqual([
                jasmine.objectContaining({ id: 'P2', name: 'Foo', description: 'I am a project' }),
                jasmine.objectContaining({ id: 'P1', name: 'Bar', description: 'I am also a project' }),
            ]);
        });

        it("should the project's experiments nested in the project", () => {
            const state = new StateBuilder()
                .clearProjects()
                .addProject({ id: 'P1', experimentIds: ['E1'] })
                .addExperiment({
                    id: 'E1',
                    name: 'Foo experiment',
                    description: 'Experimenting...',
                    duration: 123,
                    interval: 21,
                })
                .build();
            expect(selectors.getProjects(state)).toEqual([
                jasmine.objectContaining({
                    experiments: [
                        jasmine.objectContaining({
                            id: 'E1',
                            name: 'Foo experiment',
                            description: 'Experimenting...',
                            duration: 123,
                            interval: 21,
                        }),
                    ],
                })]);
        });

        it("should return the experiment's scanner nested in the experiment", () => {
            const state = new StateBuilder()
                .clearProjects()
                .addProject({ id: 'P1', experimentIds: ['E1'] })
                .addExperiment({
                    id: 'E1', scannerId: 'S1',
                })
                .addScanner({
                    id: 'S1', name: 'Scanny', isOnline: true, isFree: false,
                })
                .build();
            expect(selectors.getProjects(state)).toEqual([
                jasmine.objectContaining({
                    experiments: [
                        jasmine.objectContaining({
                            scanner: {
                                id: 'S1',
                                name: 'Scanny',
                                power: true,
                                owned: true,
                            },
                        }),
                    ],
                })]);
        });
    });

    describe('getScanners', () => {
        it('should return an empty array if there are no scanners', () => {
            const state = new StateBuilder().clearScanners().build();
            expect(selectors.getScanners(state)).toEqual([]);
        });

        it('should return the scanner information', () => {
            const state = new StateBuilder()
                .clearScanners()
                .addScanner({
                    id: 'S01', name: 'Scanny', isOnline: true, isFree: false,
                })
                .build();
            expect(selectors.getScanners(state)).toEqual([{
                identifier: 'S01', name: 'Scanny', power: true, owned: true,
            }]);
        });

        it('should sort the scanners by name', () => {
            const state = new StateBuilder()
                .clearScanners()
                .addScanner({ id: 'S01', name: 'Foo' })
                .addScanner({ id: 'S02', name: 'Bar' })
                .addScanner({ id: 'S03', name: 'Baz' })
                .build();
            expect(selectors.getScanners(state).map(s => s.name)).toEqual([
                'Bar',
                'Baz',
                'Foo',
            ]);
        });
    });

    describe('getNewProject', () => {
        it('should return null if there is no new project', () => {
            const state = new StateBuilder().clearNewProject().build();
            expect(selectors.getNewProject(state)).toBe(null);
        });

        it('should return the name and description if there is a new project', () => {
            const state = new StateBuilder()
                .setNewProjectName('Ny project')
                .setNewProjectDescription('bla bli blu')
                .build();
            expect(selectors.getNewProject(state)).toEqual({ name: 'Ny project', description: 'bla bli blu' });
        });
    });

    describe('getNewProjectErrors', () => {
        it('should return no errors if no new project', () => {
            const state = new StateBuilder().clearNewProject().build();
            const errors = selectors.getNewProjectErrors(state);
            expect(errors.size).toEqual(0);
        });

        it('should return no errors given a valid state', () => {
            const state = new StateBuilder()
                .setNewProjectName('Foo project')
                .submitNewProject()
                .build();
            const errors = selectors.getNewProjectErrors(state);
            expect(errors.size).toEqual(0);
        });

        it('should return an error if name is empty', () => {
            const state = new StateBuilder().setNewProjectName('').build();
            const errors = selectors.getNewProjectErrors(state);
            expect(errors.get('name')).toEqual('Project name cannot be empty');
        });

        it('should return no errors if the new project is not submitted', () => {
            const state = new StateBuilder().setNewProjectName('').unsubmitNewProject().build();
            const errors = selectors.getNewProjectErrors(state);
            expect(errors.size).toEqual(0);
        });
    });

    describe('getNewExperiment', () => {
        it('should return null if there is no new experiment', () => {
            const state = new StateBuilder().clearNewExperiment().build();
            expect(selectors.getNewExperiment(state)).toBe(null);
        });

        it('should return the new experiment info if there is a new experiment', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({
                    name: 'Foo',
                    description: 'bar',
                    scannerId: 'scnr01',
                    duration: 500,
                    interval: 20,
                })
                .setNewExperimentProjectId('P123')
                .build();
            expect(selectors.getNewExperiment(state)).toEqual({
                name: 'Foo',
                description: 'bar',
                scannerId: 'scnr01',
                duration: 500,
                interval: 20,
                projectId: 'P123',
            });
        });
    });

    describe('getNewExperimentErrors', () => {
        it('should return no errors if no new experiment', () => {
            const state = new StateBuilder().clearNewExperiment().build();
            expect(selectors.getNewExperimentErrors(state).size).toEqual(0);
        });

        it('should return no errors given a valid state', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({
                    name: 'New Experiment',
                    description: 'aosenuthaoseut',
                    scannerId: 'xyz',
                    interval: 10,
                    duration: 100,
                })
                .submitNewExperiment()
                .build();
            const errors = selectors.getNewExperimentErrors(state);
            expect(errors.size).toEqual(0);
        });

        it('should return no error if the experiment is not submitted', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({ name: '' })
                .unsubmitNewExperiment()
                .build();
            const errors = selectors.getNewExperimentErrors(state);
            expect(errors.size).toEqual(0);
        });

        it('should return an error if name is empty', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({ name: '' })
                .submitNewExperiment()
                .build();
            const errors = selectors.getNewExperimentErrors(state);
            expect(errors.get('name')).toEqual('Required');
        });

        it('should return an error if scannerId is empty', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({ scannerId: '' })
                .submitNewExperiment()
                .build();
            const errors = selectors.getNewExperimentErrors(state);
            expect(errors.get('scannerId')).toEqual('Required');
        });

        it('should return an error if duration is 0', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({ duration: 0 })
                .submitNewExperiment()
                .build();
            const errors = selectors.getNewExperimentErrors(state);
            expect(errors.get('duration')).toEqual('Required');
        });

        it('should return an error if interval is 0', () => {
            const state = new StateBuilder()
                .setNewExperimentValues({ interval: 0 })
                .submitNewExperiment()
                .build();
            const errors = selectors.getNewExperimentErrors(state);
            expect(errors.get('interval')).toEqual('Required');
        });
    });
});
