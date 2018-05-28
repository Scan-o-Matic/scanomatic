import plate from './plate';

describe('qc/reducers/plate', () => {
    it('returns the initial state', () => {
        expect(plate(undefined, {})).toEqual({ number: 1 });
    });

    it('resets to initial state on PROJECT_SET', () => {
        const action = { type: 'PROJECT_SET', project: 'test.a.test' };
        expect(plate({ number: 4 }, action)).toEqual({ number: 1 });
    });

    it('sets raw curve data on CURVE_RAW_SET', () => {
        const action = {
            type: 'CURVE_RAW_SET', plate: 1, row: 2, col: 3, data: [4, 5],
        };
        expect(plate(undefined, action)).toEqual({
            number: 1,
            raw: [
                [null, null, null, null],
                [null, null, null, null],
                [null, null, null, [4, 5]],
            ],
        });
    });
});
