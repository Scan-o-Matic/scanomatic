import updateStatus from './updateStatus';

describe('/statuspage/reducers/updateStatus', () => {
    it('should return empty state initially', () => {
        expect(updateStatus(undefined, {})).toEqual({
            scanners: null,
            experiments: null,
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
            date: new Date(),
        };
        expect(updateStatus(undefined, action))
            .toEqual({ scanners: action.date, experiments: null });
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
            date: new Date(),
        };
        expect(updateStatus(undefined, action))
            .toEqual({ experiments: action.date, scanners: null });
    });
});
