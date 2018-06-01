import plate from './plate';

describe('/qc/reducers/plate', () => {
    it('returns the initial state', () => {
        expect(plate(undefined, {})).toEqual({ number: 0 });
    });

    it('resets to initial state on PROJECT_SET', () => {
        const action = { type: 'PROJECT_SET', project: 'test.a.test' };
        expect(plate({ number: 4 }, action)).toEqual({ number: 0 });
    });

    describe('PLATE_SET', () => {
        it('sets the plate number and removes curves', () => {
            const action = { type: 'PLATE_SET', plate: 3 };
            expect(plate({ number: 2, raw: [[[1, 2]]] }, action)).toEqual({ number: 3 });
        });

        it('doesnt do a thing if trying to set current plate', () => {
            const action = { type: 'PLATE_SET', plate: 3 };
            expect(plate({ number: 3, raw: [[[1, 2]]] }, action))
                .toEqual({ number: 3, raw: [[[1, 2]]] });
        });
    });

    describe('CURVE_FOCUS', () => {
        it('sets curve focus', () => {
            const action = {
                type: 'CURVE_FOCUS', plate: 0, row: 1, col: 2,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                focus: {
                    row: 1,
                    col: 2,
                },
            });
        });

        it('doesnt do a thing if wrong plate', () => {
            const action = {
                type: 'CURVE_FOCUS', plate: 2, row: 1, col: 2,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
            });
        });
    });

    describe('PINNING_SET', () => {
        it('sets the pinning', () => {
            const action = {
                type: 'PINNING_SET', plate: 0, rows: 4, cols: 31,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                pinning: {
                    rows: 4,
                    cols: 31,
                },
            });
        });

        it('doesnt set if the plate is wrong', () => {
            const action = {
                type: 'PINNING_SET', plate: 2, rows: 4, cols: 31,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
            });
        });
    });
});
