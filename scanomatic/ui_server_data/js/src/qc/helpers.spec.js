import { getUpdated2DArrayCopy } from './helpers';

describe('qc/helpers', () => {
    describe('getUpdated2DArrayCopy', () => {
        it('creates expected array', () => {
            expect(getUpdated2DArrayCopy(null, 2, 3, [4, 5])).toEqual([
                [null, null, null, null],
                [null, null, null, null],
                [null, null, null, [4, 5]],
            ]);
        });

        it('returns updated copy', () => {
            const data = [
                [null, null, null, null],
                [null, null, null, null],
                [null, null, null, [4, 5]],
            ];
            expect(getUpdated2DArrayCopy(data, 3, 0, [1, 2])).toEqual([
                [null, null, null, null],
                [null, null, null, null],
                [null, null, null, [4, 5]],
                [[1, 2], null, null, null],
            ]);
        });

        it('doesnt modify the input', () => {
            const data = [
                [null, null, null, null],
                [null, null, null, null],
                [null, null, null, [4, 5]],
            ];
            getUpdated2DArrayCopy(data, 3, 0, [1, 2]);
            expect(data).toEqual([
                [null, null, null, null],
                [null, null, null, null],
                [null, null, null, [4, 5]],
                [[1, 2], null, null, null],
            ]);
        });
    });
});
