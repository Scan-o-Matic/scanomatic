import * as selectors from './selectors';
import StateBuilder from './StateBuilder';

describe('statuspage/selectors', () => {
    describe('hasLoadedScannersAndExperiments', () => {
        it('should return false if neither has been set', () => {
            const state = new StateBuilder().build();
            expect(selectors.hasLoadedScannersAndExperiments(state))
                .toBeFalsy();
        });

        it('should return false if only scanners set', () => {
            const state = new StateBuilder()
                .populateScanners()
                .build();
            expect(selectors.hasLoadedScannersAndExperiments(state))
                .toBeFalsy();
        });

        it('should return false if only experiments set', () => {
            const state = new StateBuilder()
                .populateExperiments()
                .build();
            expect(selectors.hasLoadedScannersAndExperiments(state))
                .toBeFalsy();
        });

        it('should return true if both set', () => {
            const state = new StateBuilder()
                .populateExperiments()
                .populateScanners()
                .build();
            expect(selectors.hasLoadedScannersAndExperiments(state))
                .toBeTruthy();
        });
    });

    describe('getScanners', () => {
        it('should return empty array initially', () => {
            const state = new StateBuilder().build();
            expect(selectors.getScanners(state)).toEqual([]);
        });

        it('should return the scanners', () => {
            const state = new StateBuilder().populateScanners().build();
            expect(selectors.getScanners(state)).toEqual([
                {
                    name: 'Hollowborn Heron',
                    id: 'scanner001',
                    isOnline: true,
                },
                {
                    name: 'Eclectic Eevee',
                    id: 'scanner002',
                    isOnline: false,
                },
            ]);
        });
    });

    describe('getExperiments', () => {
        it('should return empty array initially', () => {
            const state = new StateBuilder().build();
            expect(selectors.getExperiments(state)).toEqual([]);
        });

        it('should return the experiments', () => {
            const state = new StateBuilder().populateExperiments().build();
            expect(selectors.getExperiments(state)).toEqual([
                jasmine.objectContaining({
                    name: 'A quick test',
                }),
                jasmine.objectContaining({
                    name: 'Sun and Moon',
                }),
            ]);
        });
    });
});
