import * as actions from '../../src/projects/actions';
import StateBuilder from './StateBuilder';

describe('projects/actions', () => {
    beforeEach(() => {
        jasmine.clock().install();
        jasmine.clock().mockDate();
    });

    afterEach(() => {
        jasmine.clock().uninstall();
    });

    describe('initNewProject', () => {
        it('should return a NEWPROJECT_INIT action', () => {
            expect(actions.initNewProject()).toEqual({
                type: 'NEWPROJECT_INIT',
            });
        });
    });

    describe('changeNewProject', () => {
        it('should return a NEWPROJECT_CHANGE action', () => {
            expect(actions.changeNewProject('name', 'foo')).toEqual({
                type: 'NEWPROJECT_CHANGE', field: 'name', value: 'foo',
            });
        });
    });

    describe('clearNewProject', () => {
        it('should return a NEWPROJECT_CLEAR action', () => {
            expect(actions.clearNewProject()).toEqual({
                type: 'NEWPROJECT_CLEAR',
            });
        });
    });

    describe('addProject', () => {
        it('should return a PROJECTS_ADD action with an id based on the current date', () => {
            expect(actions.addProject('Some Project', 'Bla bla bla')).toEqual({
                type: 'PROJECTS_ADD',
                id: new Date().getTime().toString(),
                name: 'Some Project',
                description: 'Bla bla bla',
            });
        });
    });

    describe('initNewExperiment', () => {
        it('should return a NEWEXPERIMENT_INIT action', () => {
            expect(actions.initNewExperiment('P001')).toEqual({
                type: 'NEWEXPERIMENT_INIT',
                projectId: 'P001',
            });
        });
    });

    describe('changeNewExperiment', () => {
        it('should return a NEWEXPERIMENT_CHANGE action for string-based fields', () => {
            expect(actions.changeNewExperiment('name', 'Foobar')).toEqual({
                type: 'NEWEXPERIMENT_CHANGE',
                field: 'name',
                value: 'Foobar',
            });
        });

        it('should throw for non-string type on string-type fields', () => {
            expect(() => actions.changeNewExperiment('name', 555))
                .toThrow(new TypeError('Invalid type number for field name'));
        });

        it('should return a NEWEXPERIMENT_CHANGE action for number-based fields', () => {
            expect(actions.changeNewExperiment('duration', 1000)).toEqual({
                type: 'NEWEXPERIMENT_CHANGE',
                field: 'duration',
                value: 1000,
            });
        });

        it('should throw for non-number type on number-type fields', () => {
            expect(() => actions.changeNewExperiment('duration', new Date()))
                .toThrow(new TypeError('Invalid type object for field duration'));
        });

        it('should return a NEWEXPERIMENT_CHANGE action for object-based fields', () => {
            expect(actions.changeNewExperiment('pinning', new Map([[1, 'test']]))).toEqual({
                type: 'NEWEXPERIMENT_CHANGE',
                field: 'pinning',
                value: new Map([[1, 'test']]),
            });
        });

        it('should throw for non-object type on object-type fields', () => {
            expect(() => actions.changeNewExperiment('pinning', 5))
                .toThrow(new TypeError('Invalid type number for field pinning'));
        });
    });

    describe('clearNewExperiment', () => {
        it('should return a NEWEXPERIMENT_CLEAR action', () => {
            expect(actions.clearNewExperiment()).toEqual({
                type: 'NEWEXPERIMENT_CLEAR',
            });
        });
    });

    describe('addExperiment', () => {
        it('should return a EXPERIMENTS_ADD action with an id based on the current date', () => {
            expect(actions.addExperiment(
                'p001',
                'Some Experiment',
                'Bla bla bla',
                300,
                60,
                'sc042',
                new Map([[1, '6144']]),
            )).toEqual({
                type: 'EXPERIMENTS_ADD',
                id: new Date().getTime().toString(),
                projectId: 'p001',
                name: 'Some Experiment',
                description: 'Bla bla bla',
                duration: 300,
                interval: 60,
                scannerId: 'sc042',
                pinning: new Map([[1, '6144']]),
            });
        });
    });

    describe('startExperiment', () => {
        it('should return an EXPERIMENTS_START action', () => {
            expect(actions.startExperiment('123')).toEqual({
                type: 'EXPERIMENTS_START',
                id: '123',
                date: new Date(),
            });
        });
    });

    describe('stopExperiment', () => {
        it('should return an EXPERIMENTS_STOP action', () => {
            expect(actions.stopExperiment('123', 'mymy')).toEqual({
                type: 'EXPERIMENTS_STOP',
                id: '123',
                date: new Date(),
                reason: 'mymy',
            });
        });
    });

    describe('removeExperiment', () => {
        it('should return an EXPERIMENTS_REMOVE action', () => {
            expect(actions.removeExperiment('123')).toEqual({
                type: 'EXPERIMENTS_REMOVE',
                id: '123',
            });
        });
    });

    describe('doneExperiment', () => {
        it('should return an EXPERIMENTS_DONE action', () => {
            expect(actions.doneExperiment('123')).toEqual({
                type: 'EXPERIMENTS_DONE',
                id: '123',
            });
        });
    });

    describe('reopenExperiment', () => {
        it('should return an EXPERIMENTS_REOPEN action', () => {
            expect(actions.reopenExperiment('123')).toEqual({
                type: 'EXPERIMENTS_REOPEN',
                id: '123',
            });
        });
    });

    describe('submitNewProject', () => {
        const getState = jasmine.createSpy('getState').and.returnValue(new StateBuilder().build());

        it('should dispatch a NEWPROJECT_SUBMIT action', () => {
            const dispatch = jasmine.createSpy('dispatch');
            actions.submitNewProject()(dispatch, getState);
            expect(dispatch).toHaveBeenCalledWith({ type: 'NEWPROJECT_SUBMIT' });
        });

        describe('if new project has no errors', () => {
            beforeEach(() => {
                getState.and.returnValue(new StateBuilder()
                    .setNewProjectName('New Project')
                    .setNewProjectDescription('abcd')
                    .submitNewProject()
                    .build());
            });

            it('should dispatch a PROJECTS_ADD action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewProject()(dispatch, getState);
                expect(dispatch).toHaveBeenCalledWith(jasmine.objectContaining({ type: 'PROJECTS_ADD', name: 'New Project', description: 'abcd' }));
            });

            it('should dispatch a NEWPROJECT_CLEAR action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewProject()(dispatch, getState);
                expect(dispatch).toHaveBeenCalledWith({ type: 'NEWPROJECT_CLEAR' });
            });
        });

        describe('if new project has errors', () => {
            beforeEach(() => {
                getState.and.returnValue(new StateBuilder()
                    .setNewProjectName('')
                    .submitNewProject()
                    .build());
            });

            it('should not dispatch a PROJECTS_ADD action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewProject()(dispatch, getState);
                expect(dispatch)
                    .not.toHaveBeenCalledWith(jasmine.objectContaining({
                        type: 'PROJECTS_ADD',
                    }));
            });

            it('should not dispatch a NEWPROJECT_CLEAR action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewProject()(dispatch, getState);
                expect(dispatch)
                    .not.toHaveBeenCalledWith(jasmine.objectContaining({
                        type: 'NEWPROJECT_CLEAR',
                    }));
            });
        });
    });

    describe('submitNewExperiment', () => {
        const getState = jasmine.createSpy('getState').and.returnValue(new StateBuilder().build());

        it('should dispatch a NEWEXPERIMENT_SUBMIT action', () => {
            const dispatch = jasmine.createSpy('dispatch');
            actions.submitNewExperiment()(dispatch, getState);
            expect(dispatch).toHaveBeenCalledWith({ type: 'NEWEXPERIMENT_SUBMIT' });
        });

        describe('if new experiment has no errors', () => {
            beforeEach(() => {
                getState.and.returnValue(new StateBuilder()
                    .setNewExperimentValues({
                        name: 'Some experiment',
                        scannerId: 'xyz',
                        duration: 5000,
                        interval: 500,
                        description: 'bla bla bla',
                    })
                    .setNewExperimentProjectId('P0')
                    .submitNewExperiment()
                    .build());
            });

            it('should dispatch a EXPERIMENTS_ADD action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewExperiment()(dispatch, getState);
                expect(dispatch).toHaveBeenCalledWith(jasmine.objectContaining({
                    type: 'EXPERIMENTS_ADD',
                    name: 'Some experiment',
                    scannerId: 'xyz',
                    duration: 5000,
                    interval: 500,
                    description: 'bla bla bla',
                    projectId: 'P0',
                    pinning: new Map([[1, '384']]),
                }));
            });

            it('should dispatch a NEWEXPERIMENT_CLEAR action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewExperiment()(dispatch, getState);
                expect(dispatch).toHaveBeenCalledWith({ type: 'NEWEXPERIMENT_CLEAR' });
            });
        });

        describe('if new experiment has errors', () => {
            beforeEach(() => {
                getState.and.returnValue(new StateBuilder()
                    .setNewExperimentValues({ name: '' })
                    .submitNewExperiment()
                    .build());
            });

            it('should not dispatch a EXPERIMENTS_ADD action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewExperiment()(dispatch, getState);
                expect(dispatch).not.toHaveBeenCalledWith(jasmine.objectContaining({ type: 'EXPERIMENTS_ADD' }));
            });

            it('should dispatch a NEWEXPERIMENT_CLEAR action', () => {
                const dispatch = jasmine.createSpy('dispatch');
                actions.submitNewExperiment()(dispatch, getState);
                expect(dispatch).not.toHaveBeenCalledWith({ type: 'NEWEXPERIMENT_CLEAR' });
            });
        });
    });
});
