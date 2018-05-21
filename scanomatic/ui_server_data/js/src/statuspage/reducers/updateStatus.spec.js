import updateStatus from './updateStatus';

describe('/statuspage/reducers/updateStatus', () => {
    it('should return empty state initially', () => {
        expect(updateStatus(undefined, {})).toEqual({
            scanners: false,
            experiments: false,
        });
    });

    it('should handle SCANNERS_SET', () => {
        const action = {
            type: 'SCANNERS_SET',
            scanners: [
                {
                    name: 'Hollowborn Heron',
                    id: 'scanner001',
                    isOnline: true,
                },
            ],
        };
        expect(updateStatus(undefined, action))
            .toEqual({ scanners: true, experiments: false });
    });

    it('should handle EXPERIMENTS_SET', () => {
        const action = {
            type: 'EXPERIMENTS_SET',
            experiments: [
                {
                    id: 'job0001',
                    name: 'A quick test',
                    scannerId: 'scanner001',
                    started: new Date(),
                },
            ],
        };
        expect(updateStatus(undefined, action))
            .toEqual({ experiments: true, scanners: false });
    });
});
