import * as actions from './actions';

describe('statuspage/actions', () => {
    beforeEach(() => {
        jasmine.clock().install();
        jasmine.clock().mockDate();
    });

    afterEach(() => {
        jasmine.clock().uninstall();
    });

    it('should return a SCANNERS_SET action', () => {
        const scanners = [
            {
                name: 'Hollowborn Heron',
                id: 'scanner001',
                isOnline: true,
                isFree: true,
            },
        ];
        expect(actions.setScanners(scanners)).toEqual({
            type: 'SCANNERS_SET',
            scanners,
            date: new Date(),
        });
    });

    it('should return an EXPERIMENTS_SET action', () => {
        const experiments = [
            {
                name: 'A quick test',
                description: 'Manual testing FTW!',
                duration: 55,
                interval: 1,
                scannerId: 'scanner001',
                started: new Date(),
            },
        ];
        expect(actions.setExperiments(experiments)).toEqual({
            type: 'EXPERIMENTS_SET',
            experiments,
            date: new Date(),
        });
    });
});
