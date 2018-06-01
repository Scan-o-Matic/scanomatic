import * as actions from './actions';
import * as API from './api';
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

    describe('setPinning', () => {
        it('should return a PINNING_SET action', () => {
            expect(actions.setPinning(1, 2, 3)).toEqual({
                type: 'PINNING_SET',
                plate: 1,
                rows: 2,
                cols: 3,
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
        const plateGrowthData = {
            times: [1, 2, 3],
            raw: [[[5, 5, 5]]],
            smooth: [[[6, 6, 6]]],
        };

        beforeEach(() => {
            dispatch.calls.reset();
            spyOn(API, 'getPlateGrowthData').and
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
            expect(API.getPlateGrowthData)
                .toHaveBeenCalledWith(project, 66);
        });

        it('should dispatch setPlateGrowthData and setPinning on promise resolve', (done) => {
            const state = new StateBuilder()
                .setProject('/my/little/experiment')
                .setPlate(66)
                .build();
            const getState = () => state;
            const thunk = actions.retrievePlateCurves();
            thunk(dispatch, getState).then(() => {
                expect(dispatch)
                    .toHaveBeenCalledWith(actions.setPinning(66, 1, 1));
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
});
