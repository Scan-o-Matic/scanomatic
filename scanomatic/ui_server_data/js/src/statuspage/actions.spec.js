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
                id: 'aaa',
                name: 'A quick test',
                scannerId: 'scanner001',
                started: new Date().getTime(),
            },
        ];
        expect(actions.setExperiments(experiments)).toEqual({
            type: 'EXPERIMENTS_SET',
            experiments,
            date: new Date(),
        });
    });
});
