import * as API from '.';

describe('API', () => {
    const mostRecentRequest = () => jasmine.Ajax.requests.mostRecent();

    beforeEach(() => {
        jasmine.Ajax.install();
    });

    afterEach(() => {
        jasmine.Ajax.uninstall();
    });

    describe('setCurveQCMarkAll', () => {
        const args = [
            'my/project',
            0, // plate
            10, // row
            15, // col
            'OK',
            'myverysecretkey',
        ];

        it('should query the correct uri', () => {
            API.setCurveQCMarkAll(...args);
            expect(mostRecentRequest().url)
                .toBe('/api/results/curve_mark/set/OK/0/10/15/my/project?lock_key=myverysecretkey');
        });

        it('should return a promise that resolves on success', (done) => {
            API.setCurveQCMarkAll(...args).then(() => {
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify({ success: true }),
            });
        });

        it('should reject if responds with not success', (done) => {
            API.setCurveQCMarkAll(...args)
                .catch((e) => {
                    expect(e).toEqual(new Error('Setting QC Mark was refused'));
                    done();
                });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify({ success: false }),
            });
        });

        it('should reject on status', (done) => {
            API.setCurveQCMarkAll(...args).catch((response) => {
                expect(response).toEqual('nooo!');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400,
                responseText: JSON.stringify({ reason: 'nooo!' }),
            });
        });
    });

    describe('setCurveQCMark', () => {
        const args = [
            'my/project',
            'GenerationTime',
            0, // plate
            10, // row
            15, // col
            'BadData',
            'myverysecretkey',
        ];

        it('should query the correct uri', () => {
            API.setCurveQCMark(...args);
            expect(mostRecentRequest().url)
                .toBe('/api/results/curve_mark/set/BadData/GenerationTime/0/10/15/my/project?lock_key=myverysecretkey');
        });

        it('should return a promise that resolves on success', (done) => {
            API.setCurveQCMark(...args).then(() => {
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify({ success: true }),
            });
        });

        it('should reject if responds with not success', (done) => {
            API.setCurveQCMark(...args)
                .catch((e) => {
                    expect(e).toEqual(new Error('Setting QC Mark was refused'));
                    done();
                });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify({ success: false }),
            });
        });

        it('should reject on status', (done) => {
            API.setCurveQCMark(...args).catch((response) => {
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
