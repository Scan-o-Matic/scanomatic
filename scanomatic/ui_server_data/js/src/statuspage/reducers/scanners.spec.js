import scanners from './scanners';

describe('/statuspage/reducers/scanners', () => {
    it('should return an empty array as initial state', () => {
        expect(scanners(undefined, {})).toEqual([]);
    });

    it('should return the scanners', () => {
        const action = {
            type: 'SCANNERS_SET',
            scanners: [
                {
                    name: 'Hollowborn Heron',
                    id: 'scanner001',
                    isOnline: true,
                    isFree: true,
                },
                {
                    name: 'Eclectic Eevee',
                    id: 'scanner002',
                    isOnline: false,
                    isFree: true,
                },
            ],
            date: new Date(),
        };
        expect(scanners(undefined, action)).toEqual(action.scanners);
    });
});
