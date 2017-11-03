import { SetColonyCompressionV2 } from '../ccc/api';

describe('API', () => {
    describe('SetColonyCompressionV2', () => {
        const args = [
            {},
            'CCC42',
            '1M4G3',
            'PL4T3',
            'T0P53CR3T',
            {
                blob: [[true, false], [false, true]],
                background: [[false, true], [true, false]],
            },
            4,
            1,
            jasmine.createSpy('onSuccess'),
            jasmine.createSpy('onError'),
        ];

        beforeEach(() => {
            jasmine.Ajax.install();
        });

        afterEach(() => {
            jasmine.Ajax.uninstall();
        });

        it('should query the correct url', () => {
            SetColonyCompressionV2(...args);
            expect(jasmine.Ajax.requests.mostRecent().url)
                .toBe('/api/data/calibration/CCC42/image/1M4G3/plate/PL4T3/compress/colony/1/4');
        });
    })
});
