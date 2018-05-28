import * as selectors from './selectors';
import StateBuilder from './StateBuilder';

describe('/qc/selectors', () => {
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
            expect(selectors.getRawCurve(state, 1, 4, 4)).toBe(null);
        });

        it('should return the growth data', () => {
            const data = [1, 2, 3];
            const state = new StateBuilder().setRawCurveData(1, 4, 5, data).build();
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
            expect(selectors.getSmoothCurve(state, 1, 4, 4)).toBe(null);
        });

        it('should return the growth data', () => {
            const data = [1, 2, 3];
            const state = new StateBuilder().setSmoothCurveData(1, 4, 5, data).build();
            expect(selectors.getSmoothCurve(state, 1, 4, 5)).toEqual(data);
        });
    });
});
