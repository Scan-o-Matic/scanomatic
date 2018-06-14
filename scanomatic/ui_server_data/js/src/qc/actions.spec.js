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
});
