import * as actions from '../../src/projects/actions';

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

    describe('submitNewProject', () => {
        it('should return a NEWPROJECT_SUBMIT action', () => {
            expect(actions.submitNewProject()).toEqual({
                type: 'NEWPROJECT_SUBMIT',
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
        it('should return a NEWEXPERIMENT_CHANGE action', () => {
            expect(actions.changeNewExperiment('name', 'Foobar')).toEqual({
                type: 'NEWEXPERIMENT_CHANGE',
                field: 'name',
                value: 'Foobar',
            });
        });
    });

    describe('submitNewExperiment', () => {
        it('should return a NEWEXPERIMENT_SUBMIT action', () => {
            expect(actions.submitNewExperiment()).toEqual({
                type: 'NEWEXPERIMENT_SUBMIT',
            });
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
            )).toEqual({
                type: 'EXPERIMENTS_ADD',
                id: new Date().getTime().toString(),
                projectId: 'p001',
                name: 'Some Experiment',
                description: 'Bla bla bla',
                duration: 300,
                interval: 60,
                scanner: 'sc042',
            });
        });
    });

    describe('startExperiment', () => {
        it('should return an EXPERIMENNTS_START action', () => {
            expect(actions.startExperiment('123')).toEqual({
                type: 'EXPERIMENTS_START',
                id: '123',
                date: new Date(),
            });
        });
    });

    describe('stopExperiment', () => {
        it('should return an EXPERIMENNTS_STOP action', () => {
            expect(actions.stopExperiment('123')).toEqual({
                type: 'EXPERIMENTS_STOP',
                id: '123',
                date: new Date(),
            });
        });
    });
});
