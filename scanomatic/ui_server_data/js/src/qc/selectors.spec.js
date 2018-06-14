import * as selectors from './selectors';
import StateBuilder from './StateBuilder';

describe('/qc/selectors', () => {
    it('should get the project path', () => {
        const state = new StateBuilder().setProject('/my/path').build();
        expect(selectors.getProject(state)).toEqual('/my/path');
    });

    it('should get the plate number', () => {
        const state = new StateBuilder().setPlate(2).build();
        expect(selectors.getPlate(state)).toEqual(2);
    });

    describe('focus', () => {
        it('should return null if no focus', () => {
            const state = new StateBuilder().build();
            expect(selectors.getFocus(state)).toEqual(null);
        });

        it('should return the focused curve', () => {
            const state = new StateBuilder().setFocus(0, 2, 1).build();
            expect(selectors.getFocus(state)).toEqual({ row: 2, col: 1 });
        });
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

    describe('getCurrrentQIndexInfo', () => {
        const queue = [
            { idx: 0, col: 1, row: 1 },
            { idx: 1, col: 0, row: 1 },
            { idx: 2, col: 1, row: 0 },
            { idx: 3, col: 0, row: 0 },
        ];

        it('should return null if no queue', () => {
            const state = new StateBuilder().build();
            expect(selectors.getCurrrentQIndexInfo(state, 0)).toBe(null);
        });

        it('should return null if requesting for wrong plate', () => {
            const state = new StateBuilder()
                .setQualityIndexQueue(0, queue)
                .build();
            expect(selectors.getCurrrentQIndexInfo(state, 1)).toBe(null);
        });

        it('should return current index info', () => {
            const state = new StateBuilder()
                .setQualityIndexQueue(0, queue)
                .setQualityIndex(0, 1)
                .build();
            expect(selectors.getCurrrentQIndexInfo(state, 0)).toEqual({
                idx: 1,
                col: 0,
                row: 1,
            });
        });
    });
});
