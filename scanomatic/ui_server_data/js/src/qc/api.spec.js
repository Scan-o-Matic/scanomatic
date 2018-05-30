import * as API from './api';

describe('API (qc)', () => {
    const onSuccess = jasmine.createSpy('onSuccess');
    const onError = jasmine.createSpy('onError');
    const mostRecentRequest = () => jasmine.Ajax.requests.mostRecent();

    beforeEach(() => {
        onSuccess.calls.reset();
        onError.calls.reset();
        jasmine.Ajax.install();
    });

    afterEach(() => {
        jasmine.Ajax.uninstall();
    });

    describe('getCurveData', () => {
        const args = [
            'something/somewhere', // Project
            0, // Plate index
            10, // Row index
            5, // Col index
        ];

        it('should query the correct uri', () => {
            API.getCurveData(...args);
            expect(mostRecentRequest().url)
                .toBe('/api/results/curves/0/10/5/something/somewhere');
        });

        it('should return a promise that resolves on success', (done) => {
            API.getCurveData(...args).then((response) => {
                expect(response).toEqual({
                    project: 'something/somewhere',
                    plate: 0,
                    row: 10,
                    col: 5,
                    raw: [1, 3, 5],
                    smooth: [2, 4, 6],
                    times: [0, 1, 2],
                });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify({
                    raw_data: [1, 3, 5],
                    smooth_data: [2, 4, 6],
                    time_data: [0, 1, 2],
                }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.getCurveData(...args).catch((response) => {
                expect(response).toEqual('nooo!');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400,
                responseText: JSON.stringify({ reason: 'nooo!' }),
            });
        });
    });
});
