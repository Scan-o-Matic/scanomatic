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

    describe('pinning', () => {
        it('should get the pinning if plate is right', () => {
            const state = new StateBuilder().setPinning(2, 1).build();
            expect(selectors.getPinning(state, 0)).toEqual({ rows: 2, cols: 1 });
        });

        it('should return null if plate is wrong', () => {
            const state = new StateBuilder().setPinning(2, 1).build();
            expect(selectors.getPinning(state, 1)).toEqual(null);
        });
    });

    describe('focus', () => {
        it('should return null if no focus', () => {
            const state = new StateBuilder().build();
            expect(selectors.getFocus(state)).toEqual(null);
        });

        it('should return the focused curve', () => {
            const state = new StateBuilder().setFocus(0, 2, 1).build();
            expect(selectors.getFocus(state)).toEqual(null);
        });
    });

    describe('hasStartedLoadingPlate', () => {
        it('should be false if pinning unknown', () => {
            const state = new StateBuilder().build();
            expect(selectors.hasStartedLoadingPlate(state)).toEqual(false);
        });

        it('should be false if no curve entered', () => {
            const state = new StateBuilder().setPinning(2, 1).build();
            expect(selectors.hasStartedLoadingPlate(state)).toEqual(false);
        });

        it('should be true if curve entered', () => {
            const state = new StateBuilder()
                .setPinning(2, 1)
                .setRawCurveData(0, 0, 0, [1, 2])
                .build();
            expect(selectors.hasStartedLoadingPlate(state)).toEqual(true);
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
            const data = [1, 2, 3];
            const state = new StateBuilder()
                .setPlate(1)
                .setRawCurveData(1, 4, 5, data)
                .build();
            expect(selectors.getRawCurve(state, 1, 4, 5)).toEqual(data);
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
            const data = [1, 2, 3];
            const state = new StateBuilder()
                .setPlate(1)
                .setSmoothCurveData(1, 4, 5, data)
                .build();
            expect(selectors.getSmoothCurve(state, 1, 4, 5)).toEqual(data);
        });
    });

    describe('times', () => {
        it('should get the times of the growth curve if plate is right', () => {
            const state = new StateBuilder().setTimes(0, [1, 2, 3]);
            expect(selectors.getTimes(state, 0)).toEqual([1, 2, 3]);
        });

        it('should return null if plate is wrong', () => {
            const state = new StateBuilder().setTimes(0, [1, 2, 3]);
            expect(selectors.getTimes(state, 1)).toEqual(null);
        });
    });
});
