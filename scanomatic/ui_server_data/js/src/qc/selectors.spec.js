import * as selectors from './selectors';
import StateBuilder from './StateBuilder';

describe('/qc/selectors', () => {
    it('should get the project path', () => {
        const state = new StateBuilder().setProject('/my/path').build();
        expect(selectors.getProject(state)).toEqual('/my/path');
    });

    it('should get phenotype', () => {
        const state = new StateBuilder()
            .setProject('/my/path')
            .setPhenotype('test')
            .build();
        expect(selectors.getPhenotype(state)).toEqual('test');
    });

    it('should get phenotype data', () => {
        const state = new StateBuilder()
            .setProject('/my/path')
            .setPhenotype('test')
            .setPlatePhenotypeData(
                [[5, 4, 3], [5, 5, 1]],
                [[], []],
                [[], []],
                [[], []],
                [[], []],
            )
            .build();
        expect(selectors.getPhenotypeData(state))
            .toEqual([
                [5, 4, 3],
                [5, 5, 1],
            ]);
    });

    it('should get the QC Marks', () => {
        const state = new StateBuilder()
            .setProject('/my/path')
            .setPhenotype('test')
            .setPlatePhenotypeData(
                [[5, 4, 3], [5, 5, 1]],
                [[0], [0]],
                [[0], [1]],
                [[1, 1], [0, 1]],
                [[1, 0], [2, 2]],
            )
            .build();
        expect(selectors.getPhenotypeQCMarks(state))
            .toEqual({
                badData: [[0], [0]],
                empty: [[0], [1]],
                noGrowth: [[1, 1], [0, 1]],
                undecidedProblem: [[1, 0], [2, 2]],
            });
    });

    it('should get the plate number', () => {
        const state = new StateBuilder().setPlate(2).build();
        expect(selectors.getPlate(state)).toEqual(2);
    });

    describe('raw data', () => {
        it('should return null if curve is on wrong plate', () => {
            const state = new StateBuilder().build();
            expect(selectors.getRawCurve(state, 2, 4, 4)).toBe(null);
        });

        it('should return null if curve not yet loaded', () => {
            const state = new StateBuilder().build();
            expect(selectors.getRawCurve(state, 0, 4, 4)).toBe(null);
        });

        it('should return the growth data', () => {
            const state = new StateBuilder()
                .setPlate(1)
                .setPlateGrowthData(
                    1,
                    [1, 2, 3],
                    [[[2, 3, 4], [4, 2, 1]]],
                    [[[6, 6, 6], [1, 4, 5]]],
                )
                .build();
            expect(selectors.getRawCurve(state, 1, 0, 1)).toEqual([4, 2, 1]);
        });
    });

    describe('smooth data', () => {
        it('should return null if curve is on wrong plate', () => {
            const state = new StateBuilder().build();
            expect(selectors.getSmoothCurve(state, 2, 4, 4)).toBe(null);
        });

        it('should return null if curve not yet loaded', () => {
            const state = new StateBuilder().build();
            expect(selectors.getSmoothCurve(state, 0, 4, 4)).toBe(null);
        });

        it('should return the growth data', () => {
            const state = new StateBuilder()
                .setPlate(1)
                .setPlateGrowthData(
                    1,
                    [1, 2, 3],
                    [[[2, 3, 4], [4, 2, 1]]],
                    [[[6, 6, 6], [1, 4, 5]]],
                )
                .build();
            expect(selectors.getSmoothCurve(state, 1, 0, 1)).toEqual([1, 4, 5]);
        });
    });

    describe('times', () => {
        it('should get the times of the growth curve if plate is right', () => {
            const state = new StateBuilder()
                .setPlateGrowthData(
                    0,
                    [1, 2, 3],
                    [[[2, 3, 4]]],
                    [[[6, 6, 6]]],
                )
                .build();
            expect(selectors.getTimes(state, 0)).toEqual([1, 2, 3]);
        });

        it('should return null if plate is wrong', () => {
            const state = new StateBuilder()
                .setPlateGrowthData(
                    0,
                    [1, 2, 3],
                    [[[2, 3, 4]]],
                    [[[6, 6, 6]]],
                )
                .build();
            expect(selectors.getTimes(state, 1)).toEqual(null);
        });
    });

    describe('getFocus', () => {
        const queue = [
            { idx: 0, col: 1, row: 1 },
            { idx: 1, col: 0, row: 1 },
            { idx: 2, col: 1, row: 0 },
            { idx: 3, col: 0, row: 0 },
        ];

        it('should return null if no queue', () => {
            const state = new StateBuilder().build();
            expect(selectors.getFocus(state)).toBe(null);
        });

        it('should return current index info', () => {
            const state = new StateBuilder()
                .setQualityIndexQueue(queue)
                .setQualityIndex(1)
                .build();
            expect(selectors.getFocus(state)).toEqual({
                idx: 1,
                col: 0,
                row: 1,
            });
        });
    });

    describe('getQIndexFromPosition', () => {
        const queue = [
            { idx: 0, col: 1, row: 1 },
            { idx: 1, col: 0, row: 1 },
            { idx: 2, col: 1, row: 0 },
            { idx: 3, col: 0, row: 0 },
        ];

        it('should return null if no queue', () => {
            const state = new StateBuilder().build();
            expect(selectors.getQIndexFromPosition(state, 1, 0)).toBe(null);
        });

        it('should return index for position', () => {
            const state = new StateBuilder()
                .setQualityIndexQueue(queue)
                .setQualityIndex(1)
                .build();
            expect(selectors.getQIndexFromPosition(state, 1, 0)).toEqual(1);
        });

        it('should return undefined for unknown position', () => {
            const state = new StateBuilder()
                .setQualityIndexQueue(queue)
                .setQualityIndex(1)
                .build();
            expect(selectors.getQIndexFromPosition(state, 10, 0)).toBe(undefined);
        });
    });
});
