import * as actions from './actions';

describe('qc/actions', () => {
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

    describe('setRawCurveData', () => {
        it('should return a CURVE_RAW_SET action', () => {
            expect(actions.setRawCurveData(1, 2, 3, [4, 5, 6]))
                .toEqual({
                    type: 'CURVE_RAW_SET',
                    plate: 1,
                    row: 2,
                    col: 3,
                    data: [4, 5, 6],
                });
        });
    });

    describe('setSmoothCurveData', () => {
        it('should return a CURVE_SMOOTH_SET action', () => {
            expect(actions.setSmoothCurveData(1, 2, 3, [4, 5, 6]))
                .toEqual({
                    type: 'CURVE_SMOOTH_SET',
                    plate: 1,
                    row: 2,
                    col: 3,
                    data: [4, 5, 6],
                });
        });
    });
});
