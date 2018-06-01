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
});
