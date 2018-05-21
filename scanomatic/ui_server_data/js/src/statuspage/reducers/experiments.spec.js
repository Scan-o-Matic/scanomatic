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
                    id: 'job001',
                    name: 'A quick test',
                    scannerId: 'scanner001',
                    started: new Date().getTime(),
                },
            ],
            date: new Date(),
        };
        expect(experiments(undefined, action)).toEqual(action.experiments);
    });

    it('should filter out experiments that have not started', () => {
        const started = new Date().getTime();
        const action = {
            type: 'EXPERIMENTS_SET',
            experiments: [
                {
                    id: 'job001',
                    name: 'console.log',
                    scannerId: 'scanner001',
                    started,
                },
                {
                    id: 'job002',
                    name: 'Reading the docs',
                    scannerId: 'scanner001',
                },
            ],
            date: new Date(),
        };
        expect(experiments(undefined, action))
            .toEqual([
                {
                    id: 'job001',
                    name: 'console.log',
                    scannerId: 'scanner001',
                    started,
                },
            ]);
    });
});
