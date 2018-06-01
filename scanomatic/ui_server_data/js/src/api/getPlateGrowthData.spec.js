import * as API from '.';

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

    describe('getPlateGrowthData', () => {
        const args = [
            'something/somewhere', // Project
            0, // Plate index
        ];

        it('should query the correct uri', () => {
            API.getPlateGrowthData(...args);
            expect(mostRecentRequest().url)
                .toBe('/api/results/growthcurves/0/something/somewhere');
        });

        it('should return a promise that resolves on success', (done) => {
            // a 1x1 plate with 3 scans in the experiment, because I'm lazy
            API.getPlateGrowthData(...args).then((response) => {
                expect(response).toEqual({
                    raw: [[[1, 2, 3]]],
                    smooth: [[[2, 3, 4]]],
                    times: [0, 1, 2],
                });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify({
                    raw_data: [[[1, 2, 3]]],
                    smooth_data: [[[2, 3, 4]]],
                    times_data: [0, 1, 2],
                }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.getPlateGrowthData(...args).catch((response) => {
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
