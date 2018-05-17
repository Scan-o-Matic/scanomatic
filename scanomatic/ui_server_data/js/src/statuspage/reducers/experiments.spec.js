import experiments from './experiments';

describe('/statuspage/reducers/experiments', () => {
    it('should return empty initial state', () => {
        expect(experiments(undefined, {})).toEqual([]);
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
        expect(experiments(undefined, action)).toEqual(action.experiments);
    });

    it('should filter out experiments that have not started', () => {
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
                {
                    name: 'Planned',
                    description: 'Reading the docs',
                    duration: 55,
                    interval: 1,
                    scannerId: 'scanner001',
                },
            ],
            date: new Date(),
        };
        expect(experiments(undefined, action))
            .toEqual([
                {
                    name: 'A quick test',
                    description: 'Manual testing FTW!',
                    duration: 55,
                    interval: 1,
                    scannerId: 'scanner001',
                    started: new Date(),
                },
            ]);
    });
});
