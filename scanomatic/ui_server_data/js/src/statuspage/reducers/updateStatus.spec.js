import updateStatus from './updateStatus';

describe('/statuspage/reducers/updateStatus', () => {
    it('should return empty state initially', () => {
        expect(updateStatus(undefined, {})).toEqual({});
    });

    it('should handle SCANNERS_SET', () => {
        const action = {
            type: 'SCANNERS_SET',
            scanners: [
                {
                    name: 'Hollowborn Heron',
                    id: 'scanner001',
                    isOnline: true,
                    isFree: true,
                },
            ],
            date: new Date(),
        };
        expect(updateStatus(undefined, action)).toEqual({ scanners: action.date });
    });

    it('should handle EXPERIMENTS_SET', () => {
        const action = {
            type: 'EXPERIMENTS_SET',
            experiments: [
                {
                    name: 'A quick test',
                    description: 'Manual testing FTW!',
                    duration: 55,
                    interval: 1,
                    scannerId: 'scanner001',
                    started: new Date(),
                },
            ],
            date: new Date(),
        };
        expect(updateStatus(undefined, action)).toEqual({ experiments: action.date });
    });
});
