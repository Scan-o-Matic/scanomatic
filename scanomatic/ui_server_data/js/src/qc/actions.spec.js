import * as actions from './actions';

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

    describe('setTimes', () => {
        it('should return a TIMES_SET action', () => {
            expect(actions.setTimes(0, [1, 2, 3])).toEqual({
                type: 'TIMES_SET',
                plate: 0,
                times: [1, 2, 3],
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
