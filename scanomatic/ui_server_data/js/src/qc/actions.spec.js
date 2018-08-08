import * as actions from './actions';
import * as API from '../api';
import StateBuilder from './StateBuilder';
import FakePromise from '../helpers/FakePromise';

describe('/qc/actions', () => {
    describe('setPlate', () => {
        it('should return a PLATE_SET action', () => {
            expect(actions.setPlate(5)).toEqual({
                type: 'PLATE_SET',
                plate: 5,
            });
        });
    });

    describe('setProject', () => {
        it('should return a PROJECT_SET action', () => {
            expect(actions.setProject('test.me')).toEqual({
                type: 'PROJECT_SET',
                project: 'test.me',
            });
        });
    });

    describe('setPhenotype', () => {
        it('should return a PHENOTYPE_SET action', () => {
            expect(actions.setPhenotype('death rate')).toEqual({
                type: 'PHENOTYPE_SET',
                phenotype: 'death rate',
            });
        });
    });

    describe('setShowNormalized', () => {

    });

    describe('setPlatePhenotypeData', () => {
        it('should return a PLATE_PHENOTYPEDATA_SET action', () => {
            expect(actions.setPlatePhenotypeData(
                1,
                'GenerationTimeWhen',
                [[0]],
                new Map([
                    ['BadData', [[0], [0]]],
                    ['Empty', [[1], [1]]],
                    ['NoGrowth', [[2], [2]]],
                    ['UndecidedProblem', [[3], [3]]],
                ]),
            ))
                .toEqual({
                    type: 'PLATE_PHENOTYPEDATA_SET',
                    plate: 1,
                    phenotype: 'GenerationTimeWhen',
                    phenotypes: [[0]],
                    qcmarks: new Map([
                        ['BadData', [[0], [0]]],
                        ['Empty', [[1], [1]]],
                        ['NoGrowth', [[2], [2]]],
                        ['UndecidedProblem', [[3], [3]]],
                    ]),
                });
        });
    });

    describe('focusCurve', () => {
        it('should return a CURVE_FOCUS action', () => {
            expect(actions.focusCurve(0, 1, 2)).toEqual({
                type: 'CURVE_FOCUS',
                plate: 0,
                row: 1,
                col: 2,
            });
        });
    });

    describe('setPlateGrowthData', () => {
        it('should return a PLATE_GROWTHDATA_SET action', () => {
            const times = [1, 2, 3];
            const smooth = [[[2, 3, 4]]];
            const raw = [[[5, 4, 3]]];
            const plate = 3;

            expect(actions.setPlateGrowthData(
                plate,
                times,
                raw,
                smooth,
            ))
                .toEqual({
                    type: 'PLATE_GROWTHDATA_SET',
                    plate,
                    times,
                    raw,
                    smooth,
                });
        });
    });

    describe('updateFocusCurveQCMark', () => {
        const dispatch = jasmine.createSpy('dispatch');
        let setCurveQCMark;
        let setCurveQCMarkAll;

        beforeEach(() => {
            dispatch.calls.reset();
            setCurveQCMark = spyOn(API, 'setCurveQCMark')
                .and.callFake((project) => {
                    if (project === 'fail/me') return Promise.reject(new Error('bad idea'));
                    return FakePromise.resolve();
                });
            setCurveQCMarkAll = spyOn(API, 'setCurveQCMarkAll')
                .and.callFake((project) => {
                    if (project === 'fail/me') return Promise.reject(new Error('bad idea'));
                    return FakePromise.resolve();
                });
        });

        it('returns a function that throws error if no project', () => {
            const state = new StateBuilder().build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            expect(() => thunk(dispatch, () => state))
                .toThrow(new Error('Cant set QC Mark if no project'));
        });

        it('returns a function that throws if no focus curve', () => {
            const state = new StateBuilder()
                .setProject('my/experiment')
                .setPlate(0)
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            expect(() => thunk(dispatch, () => state))
                .toThrow(new Error('Cant set QC Mark if no focus'));
        });

        it('returns a function that throws if focus position is dirty', () => {
            const state = new StateBuilder()
                .setProject('my/experiment')
                .setPhenotype('GenerationTime')
                .setQualityIndexQueue([{ idx: 0, col: 0, row: 10 }])
                .setDirty(10, 0)
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            expect(() => thunk(dispatch, () => state))
                .toThrow(new Error('Cannot set mark while previous mark is still processing for this position'));
        });

        it('dispatches a store update immidiately', () => {
            const state = new StateBuilder()
                .setProject('my/experiment')
                .setPlate(0)
                .setQualityIndexQueue([{ idx: 0, col: 0, row: 10 }])
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            thunk(dispatch, () => state);
            expect(dispatch)
                .toHaveBeenCalledWith(actions.setStoreCurveQCMarkDirty(0, 10, 0, 'OK', 'GenerationTime'));
        });

        it('calls API.setCurveQCMark if phenotype supplied', () => {
            const state = new StateBuilder()
                .setProject('my/experiment')
                .setPlate(0)
                .setQualityIndexQueue([{ idx: 0, col: 0, row: 10 }])
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            thunk(dispatch, () => state);
            expect(setCurveQCMark)
                .toHaveBeenCalledWith('my/experiment', 0, 10, 0, 'OK', 'GenerationTime', 'letmedothis');
        });

        it('calls API.setCurveQCMarkAll if no phenotype specified', () => {
            const state = new StateBuilder()
                .setProject('my/experiment')
                .setPlate(0)
                .setQualityIndexQueue([{ idx: 0, col: 0, row: 10 }])
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', null, 'letmedothis');
            thunk(dispatch, () => state);
            expect(setCurveQCMarkAll)
                .toHaveBeenCalledWith('my/experiment', 0, 10, 0, 'OK', 'letmedothis');
        });

        it('dispatches a reverting of mark if API request fails when setting', (done) => {
            const state = new StateBuilder()
                .setProject('fail/me')
                .setPlate(2)
                .setPhenotype('GenerationTime')
                .setQualityIndexQueue([{ idx: 0, row: 0, col: 1 }])
                .setPlatePhenotypeData(
                    'GenerationTime',
                    [[10, 154]],
                    new Map([
                        ['BadData', [[0], [1]]],
                        ['Empty', null],
                    ]),
                )
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            thunk(dispatch, () => state)
                .then(() => {
                    expect(dispatch.calls.count()).toBe(2);
                    expect(dispatch.calls.argsFor(0))
                        .toEqual([actions.setStoreCurveQCMarkDirty(2, 0, 1, 'OK', 'GenerationTime')]);
                    expect(dispatch.calls.argsFor(1))
                        .toEqual([actions.setStoreCurveQCMark(2, 0, 1, 'BadData', 'GenerationTime')]);
                    done();
                });
        });

        it('dispatches a remove dirty if API request goes through', (done) => {
            const state = new StateBuilder()
                .setProject('my/experiment')
                .setPlate(2)
                .setPhenotype('GenerationTime')
                .setQualityIndexQueue([{ idx: 0, row: 0, col: 1 }])
                .setPlatePhenotypeData(
                    'GenerationTime',
                    [[10, 154]],
                    new Map([
                        ['BadData', [[0], [1]]],
                        ['Empty', null],
                    ]),
                )
                .build();
            const thunk = actions.updateFocusCurveQCMark('OK', 'GenerationTime', 'letmedothis');
            thunk(dispatch, () => state)
                .then(() => {
                    expect(dispatch.calls.argsFor(0))
                        .toEqual([actions.setStoreCurveQCMarkDirty(2, 0, 1, 'OK', 'GenerationTime')]);
                    expect(dispatch.calls.argsFor(1))
                        .toEqual([actions.setQCMarkNotDirty(2, 0, 1)]);
                    expect(dispatch.calls.count()).toBe(2);
                    done();
                });
        });
    });

    describe('retrievePlateCurves ThunkAction', () => {
        const dispatch = jasmine.createSpy('dispatch');
        let getPlateGrowthData;
        const plateGrowthData = {
            times: [1, 2, 3],
            raw: [[[5, 5, 5]]],
            smooth: [[[6, 6, 6]]],
        };

        beforeEach(() => {
            dispatch.calls.reset();
            getPlateGrowthData = spyOn(API, 'getPlateGrowthData').and
                .returnValue(FakePromise.resolve(plateGrowthData));
        });

        it('returns a function that throws error if no project', () => {
            const state = new StateBuilder().build();
            const getState = () => state;
            const thunk = actions.retrievePlateCurves();
            expect(() => thunk(dispatch, getState))
                .toThrow(new Error('Cannot retrieve curves if project not set'));
        });

        it('should call API.getPlateGrowthData with correct params', () => {
            const project = '/my/little/experiment';
            const state = new StateBuilder()
                .setProject(project)
                .setPlate(66)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlateCurves();
            thunk(dispatch, getState);
            expect(getPlateGrowthData)
                .toHaveBeenCalledWith(project, 66);
        });

        it('should dispatch setPlateGrowthData on promise resolve', (done) => {
            const state = new StateBuilder()
                .setProject('/my/little/experiment')
                .setPlate(66)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlateCurves();
            thunk(dispatch, getState).then(() => {
                expect(dispatch)
                    .toHaveBeenCalledWith(actions.setPlateGrowthData(
                        66,
                        plateGrowthData.times,
                        plateGrowthData.raw,
                        plateGrowthData.smooth,
                    ));
                done();
            });
        });
    });

    describe('retrievePhenotypesNeededInGraph', () => {
        const dispatch = jasmine.createSpy('dispatch');
        const gtData = {
            phenotypes: [[0]],
            qcmarks: new Map([
                ['BadData', [[], []]],
                ['Empty', [[0], [1]]],
                ['NoGrowth', [[1], [0]]],
                ['UndecidedProblem', [[1, 1], [0, 1]]],
            ]),
        };
        const gtWhenData = {
            phenotypes: [[5]],
            qcmarks: new Map([
                ['BadData', [[1], [1]]],
                ['Empty', [[2], [1]]],
                ['NoGrowth', [[1], [2]]],
                ['UndecidedProblem', [[2, 1], [0, 1]]],
            ]),
        };
        const expYeildData = {
            phenotypes: [[4]],
            qcmarks: new Map([
                ['BadData', [[1], [4]]],
                ['Empty', [[2], [4]]],
                ['NoGrowth', [[1], [4]]],
                ['UndecidedProblem', [[2, 1], [0, 4]]],
            ]),
        };
        let getPhenotypeData;

        beforeEach(() => {
            dispatch.calls.reset();
            getPhenotypeData = spyOn(API, 'getPhenotypeData')
                .and.callFake((project, plate, phenotype) => FakePromise.resolve({
                    GenerationTime: gtData,
                    GenerationTimeWhen: gtWhenData,
                    ExperimentGrowthYield: expYeildData,
                }[phenotype]));
        });

        it('returns a function that throws error if no project', () => {
            const state = new StateBuilder()
                .build();
            const getState = () => state;
            const thunk = actions.retrievePhenotypesNeededInGraph(0);
            expect(() => thunk(dispatch, getState))
                .toThrow(new Error('Cannot retrieve phenotype if project not set'));
        });

        it('returns a function that returns resolved promise without calling the api if plate missmatch', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .build();
            const getState = () => state;
            const thunk = actions.retrievePhenotypesNeededInGraph(1);
            thunk(dispatch, getState)
                .then(() => {
                    expect(dispatch).not.toHaveBeenCalled();
                    expect(getPhenotypeData).not.toHaveBeenCalled();
                    done();
                });
        });

        it('fetches GenerationTime, GenerationTimeWhen & ExperimentGrowthYield', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .build();
            const getState = () => state;
            const thunk = actions.retrievePhenotypesNeededInGraph(0);
            thunk(dispatch, getState)
                .then(() => {
                    expect(getPhenotypeData).toHaveBeenCalledWith('my/project', 0, 'GenerationTime', false);
                    expect(getPhenotypeData).toHaveBeenCalledWith('my/project', 0, 'GenerationTimeWhen', false);
                    expect(getPhenotypeData).toHaveBeenCalledWith('my/project', 0, 'ExperimentGrowthYield', false);
                    done();
                });
        });

        it('skips those phenoypes it already has', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .setPlatePhenotypeData('GenerationTime', gtData.phenotypes, gtData.qcmarks)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePhenotypesNeededInGraph(0);
            thunk(dispatch, getState)
                .then(() => {
                    expect(getPhenotypeData).not.toHaveBeenCalledWith('my/project', 0, 'GenerationTime', false);
                    expect(getPhenotypeData).toHaveBeenCalledWith('my/project', 0, 'GenerationTimeWhen', false);
                    expect(getPhenotypeData).toHaveBeenCalledWith('my/project', 0, 'ExperimentGrowthYield', false);
                    done();
                });
        });

        it('dispatches updates of phenotypes and qcmarks for all fetched phenotypes', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .setPlatePhenotypeData('ExperimentGrowthYield', expYeildData.phenotypes, expYeildData.qcmarks)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePhenotypesNeededInGraph(0);
            thunk(dispatch, getState)
                .then(() => {
                    expect(dispatch).toHaveBeenCalledWith(actions.setPlatePhenotypeData(
                        0,
                        'GenerationTime',
                        gtData.phenotypes,
                        gtData.qcmarks,
                    ));
                    expect(dispatch).toHaveBeenCalledWith(actions.setPlatePhenotypeData(
                        0,
                        'GenerationTimeWhen',
                        gtWhenData.phenotypes,
                        gtWhenData.qcmarks,
                    ));
                    expect(dispatch.calls.count()).toEqual(2);
                    done();
                });
        });
    });

    describe('retrievePlatePhenotype', () => {
        const dispatch = jasmine.createSpy('dispatch');
        const gtData = {
            phenotypes: [[0]],
            qcmarks: new Map([
                ['BadData', [[], []]],
                ['Empty', [[0], [1]]],
                ['NoGrowth', [[1], [0]]],
                ['UndecidedProblem', [[1, 1], [0, 1]]],
            ]),
        };
        let getPhenotypeData;

        beforeEach(() => {
            dispatch.calls.reset();
            getPhenotypeData = spyOn(API, 'getPhenotypeData')
                .and.returnValue(FakePromise.resolve(gtData));
        });

        it('returns a function that throws error if no project', () => {
            const state = new StateBuilder()
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlatePhenotype(0);
            expect(() => thunk(dispatch, getState))
                .toThrow(new Error('Cannot retrieve phenotype if project not set'));
        });

        it('returns a function that throws if no phenotype set', () => {
            const state = new StateBuilder()
                .setProject('my/project')
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlatePhenotype(0);
            expect(() => thunk(dispatch, getState))
                .toThrow(new Error('Cannot retrieve phenotype if phenotype not set'));
        });

        it('returns a resolved promise without calling the api if already has the data', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .setPhenotype('GenerationTime')
                .setPlatePhenotypeData('GenerationTime', gtData.phenotypes)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlatePhenotype(0);
            thunk(dispatch, getState)
                .then(() => {
                    expect(dispatch).not.toHaveBeenCalled();
                    expect(getPhenotypeData).not.toHaveBeenCalled();
                    done();
                });
        });

        it('changes plate if requesting to retrieve for other than current plate', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .setPhenotype('GenerationTime')
                .setPlatePhenotypeData('GenerationTime', gtData.phenotypes, gtData.qcmarks)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlatePhenotype(1);
            thunk(dispatch, getState)
                .then(() => {
                    expect(dispatch).toHaveBeenCalledWith(actions.setPlate(1));
                    done();
                });
        });

        it('doesnt change plate if not needed', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .setPhenotype('GenerationTime')
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlatePhenotype(0);
            thunk(dispatch, getState)
                .then(() => {
                    expect(dispatch).not.toHaveBeenCalledWith(actions.setPlate(0));
                    done();
                });
        });

        it('retrieves the phenotype data and dispatches updates', (done) => {
            const state = new StateBuilder()
                .setProject('my/project')
                .setPhenotype('GenerationTime')
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlatePhenotype(0);
            thunk(dispatch, getState)
                .then(() => {
                    expect(getPhenotypeData).toHaveBeenCalledWith('my/project', 0, 'GenerationTime', false);
                    expect(dispatch).toHaveBeenCalledWith(actions.setPlatePhenotypeData(
                        0,
                        'GenerationTime',
                        gtData.phenotypes,
                        gtData.qcmarks,
                    ));
                    done();
                });
        });
    });

    describe('quality index', () => {
        it('setQualityIndexQueue should return a QUALITYINDEX_QUEUE_SET acation', () => {
            const queue = [{ idx: 0, col: 4, row: 10 }, { idx: 1, col: 2, row: 55 }];
            expect(actions.setQualityIndexQueue(queue)).toEqual({
                type: 'QUALITYINDEX_QUEUE_SET',
                queue,
            });
        });

        it('setQualityIndex should return a QUALITYINDEX_SET action', () => {
            const index = 42;
            expect(actions.setQualityIndex(index)).toEqual({
                type: 'QUALITYINDEX_SET',
                index,
            });
        });

        it('nextQualityIndex should return a QUALITYINDEX_NEXT action', () => {
            expect(actions.nextQualityIndex()).toEqual({
                type: 'QUALITYINDEX_NEXT',
            });
        });

        it('previousQualityIndex should return a QUALITYINDEX_PREVIOUS action', () => {
            expect(actions.previousQualityIndex()).toEqual({
                type: 'QUALITYINDEX_PREVIOUS',
            });
        });
    });

    describe('setStoreCurveQCMark', () => {
        it('should create a CURVE_QCMARK_SET action for all phenotypes', () => {
            expect(actions.setStoreCurveQCMark(
                0,
                1,
                2,
                'OK',
            ))
                .toEqual({
                    type: 'CURVE_QCMARK_SET',
                    phenotype: undefined,
                    mark: 'OK',
                    plate: 0,
                    row: 1,
                    col: 2,
                    dirty: false,
                });
        });

        it('should create a CURVE_QCMARK_SET action for GenerationTime', () => {
            expect(actions.setStoreCurveQCMark(
                0,
                1,
                2,
                'OK',
                'GenerationTime',
            ))
                .toEqual({
                    type: 'CURVE_QCMARK_SET',
                    phenotype: 'GenerationTime',
                    mark: 'OK',
                    plate: 0,
                    row: 1,
                    col: 2,
                    dirty: false,
                });
        });
    });

    describe('setStoreCurveQCMarkDirty', () => {
        it('should create a CURVE_QCMARK_SETDIRTY action for GenerationTime', () => {
            expect(actions.setStoreCurveQCMarkDirty(
                0,
                1,
                2,
                'OK',
                'GenerationTime',
            ))
                .toEqual({
                    type: 'CURVE_QCMARK_SET',
                    phenotype: 'GenerationTime',
                    mark: 'OK',
                    plate: 0,
                    row: 1,
                    col: 2,
                    dirty: true,
                });
        });
    });

    describe('setQCMarkNotDirty', () => {
        it('should create a CURVE_QCMARK_REMOVEDIRTY action', () => {
            expect(actions.setQCMarkNotDirty(0, 1, 2))
                .toEqual({
                    type: 'CURVE_QCMARK_REMOVEDIRTY',
                    plate: 0,
                    row: 1,
                    col: 2,
                });
        });
    });
});
