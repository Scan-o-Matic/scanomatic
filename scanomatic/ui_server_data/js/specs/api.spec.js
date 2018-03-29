import 'jasmine-ajax';

import * as API from '../src/api';
import Duration from '../src/Duration';

const toHaveMethod = (util, customEqualityTesters) => ({
    compare: (request, expected) => {
        const pass = util.equals(
            request.method.toUpperCase(),
            expected.toUpperCase(),
            customEqualityTesters,
        );
        return { pass };
    },
});

describe('API', () => {
    const onSuccess = jasmine.createSpy('onSuccess');
    const onError = jasmine.createSpy('onError');
    const mostRecentRequest = () => jasmine.Ajax.requests.mostRecent();

    beforeEach(() => {
        onSuccess.calls.reset();
        onError.calls.reset();
        jasmine.Ajax.install();
        jasmine.addMatchers({ toHaveMethod });
    });

    afterEach(() => {
        jasmine.Ajax.uninstall();
    });

    describe('SetGridding', () => {
        const cccId = 'CCC42';
        const imageId = '1M4G3';
        const plate = 0;
        const pinningFormat = [42, 18];
        const offset = [6, 7];
        const accessToken = 'open for me';
        const successCallback = jasmine.createSpy('success');
        const errorCallback = jasmine.createSpy('error');
        const args = [
            cccId, imageId, plate, pinningFormat, offset, accessToken,
            successCallback, errorCallback,
        ];

        beforeEach(() => {
            successCallback.calls.reset();
            errorCallback.calls.reset();
        });

        it('should query the correct url', () => {
            API.SetGridding(...args);
            expect(mostRecentRequest().url)
                .toBe('/api/calibration/CCC42/image/1M4G3/plate/0/grid/set');
        });

        it('should send a POST request', () => {
            API.SetGridding(...args);
            expect(mostRecentRequest()).toHaveMethod('POST');
        });

        it('should send the pinning format', ()=>{
            API.SetGridding(...args);
            const params = JSON.parse(mostRecentRequest().params);
            expect(params.pinning_format).toEqual(pinningFormat);
        });

        it('should send the offset', ()=>{
            API.SetGridding(...args);
            const params = JSON.parse(mostRecentRequest().params);
            expect(params.gridding_correction).toEqual(offset);
        });

        it('should send the access token', ()=>{
            API.SetGridding(...args);
            const params = JSON.parse(mostRecentRequest().params);
            expect(params.access_token).toEqual(accessToken);
        });

        it('Should return a promise that resolves on success', (done) => {
            API.SetGridding(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            const errorData = { reason: '(+_+)', grid: [] };
            API.SetGridding(...args).catch(data => {
                expect(data).toEqual(errorData);
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify(errorData),
            });
        });
    });

    describe('SetColonyCompression', () => {
        const cellCount = 666;
        const args = [
            'CCC42',
            '1M4G3',
            'PL4T3',
            'T0P53CR3T',
            {
                blob: [[true, false], [false, true]],
                background: [[false, true], [true, false]],
            },
            cellCount,
            4,
            1,
            onSuccess,
            onError,
        ];

        beforeEach(() => {
            onSuccess.calls.reset();
            onError.calls.reset();
        });

        it('should query the correct url', () => {
            API.SetColonyCompression(...args);
            expect(jasmine.Ajax.requests.mostRecent().url)
                .toBe('/api/calibration/CCC42/image/1M4G3/plate/PL4T3/compress/colony/1/4');
        });

        it('should send the cell count', () => {
            API.SetColonyCompression(...args);
            const params = JSON.parse(mostRecentRequest().params);
            expect(params.cell_count).toEqual(cellCount);
        });

        it('should call onSuccess on success', () => {
            const data = { foo: 'bar' };
            API.SetColonyCompression(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 200, responseText: JSON.stringify(data),
            });
            expect(onSuccess).toHaveBeenCalledWith(data);
        });

        it('should call onError on error', () => {
            const data = { foo: 'bar' };
            API.SetColonyCompression(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 400, responseText: JSON.stringify(data),
            });
            expect(onError).toHaveBeenCalledWith(data);
        });
    })

    describe('SetColonyDetection', () => {
        const args = [
            'CCC42',
            '1M4G3',
            'PL4T3',
            'T0P53CR3T',
            4,
            1,
            onSuccess,
            onError,
        ];

        beforeEach(() => {
            onSuccess.calls.reset();
            onError.calls.reset();
        });

        it('should query the correct url', () => {
            API.SetColonyDetection(...args);
            expect(jasmine.Ajax.requests.mostRecent().url)
                .toBe('/api/calibration/CCC42/image/1M4G3/plate/PL4T3/detect/colony/1/4');
        });

        it('should call onSuccess on success', () => {
            const data = { foo: 'bar' };
            API.SetColonyDetection(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 200, responseText: JSON.stringify(data),
            });
            expect(onSuccess).toHaveBeenCalledWith(data);
        });

        it('should call onError on error', () => {
            const data = { foo: 'bar' };
            API.SetColonyDetection(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 400, responseText: JSON.stringify(data),
            });
            expect(onError).toHaveBeenCalledWith(data);
        });
    })

    describe('GetMarkers', () => {
        const image = new File(['foo'], 'myimage.tiff');
        const args = [
            'MyFixture123',
            image,
        ];

        const mostRecentUrl = () => jasmine.Ajax.requests.mostRecent().url;

        it('should query the correct url', () => {
            API.GetMarkers(...args)
            expect(mostRecentUrl()).toBe('/api/data/markers/detect/MyFixture123');
        });

        it('should send a POST request', () => {
            API.GetMarkers(...args)
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should send the file', () => {
            API.GetMarkers(...args)
            expect(mostRecentRequest().params.get('image')).toEqual(image);
        });

        it('should set "save" to false', () => {
            API.GetMarkers(...args)
            expect(mostRecentRequest().params.get('save')).toEqual('false');
        });

        it('should return a promise that resolve on success', (done) => {
            API.GetMarkers(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.GetMarkers(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('GetImageid', () => {
        const image = new File(['foo'], 'myimage.tiff');
        const args = [
            'CCC0',
            image,
            'T0K3N',
        ];

        it('should query the correct url', () => {
            API.GetImageId(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/add_image');
        });

        it('should send a POST request', () => {
            API.GetImageId(...args);
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should send the image', () => {
            API.GetImageId(...args);
            expect(mostRecentRequest().params.get('image')).toEqual(image);
        });

        it('should send the access token', () => {
            API.GetImageId(...args);
            expect(mostRecentRequest().params.get('access_token')).toEqual('T0K3N');
        });

        it('should return a promise that resolves on success', (done) => {
            API.GetImageId(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.GetImageId(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('SetCccImageData', () => {
        const args = [
            'CCC0',
            'IMG0',
            'T0K3N',
            [{ key: 'key1', value: 'value1' }, { key: 'key2', value: 'value2' }],
            'MyFixture',
        ];

        it('should query the correct url', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/image/IMG0/data/set');
        });

        it('should send a POST request', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should send the access token', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().params.get('access_token'))
                .toEqual('T0K3N');
        });

        it('should send the ccc id', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().params.get('ccc_identifier'))
                .toEqual('CCC0');
        });

        it('should send the image id', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().params.get('image_identifier'))
                .toEqual('IMG0');
        });

        it('should send the fixture name', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().params.get('fixture'))
                .toEqual('MyFixture');
        });

        it('should send the passed in data', () => {
            API.SetCccImageData(...args);
            expect(mostRecentRequest().params.get('key1'))
                .toEqual('value1');
            expect(mostRecentRequest().params.get('key2'))
                .toEqual('value2');
        });

        it('should return a promise that resolves on success', (done) => {
            API.SetCccImageData(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.SetCccImageData(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('SetCccImageSlice', () => {
        const args = [
            'CCC0',
            'IMG0',
            'T0K3N',
        ];

        it('should query the correct url', () => {
            API.SetCccImageSlice(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/image/IMG0/slice/set');
        });

        it('should send a POST request', () => {
            API.SetCccImageSlice(...args);
            expect(mostRecentRequest()).toHaveMethod('POST');
        });

        it('should send the access token', () => {
            API.SetCccImageSlice(...args);
            expect(mostRecentRequest().params.get('access_token'))
                .toEqual('T0K3N');
        });

        it('should return a promise that resolves on success', () => {
            API.SetCccImageSlice(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', () => {
            API.SetCccImageSlice(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('SetGrayScaleImageAnalysis', () => {
        const args = [
            'CCC0',
            'IMG0',
            'T0K3N',
        ];

        it('should query the correct URL', () => {
            API.SetGrayScaleImageAnalysis(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/image/IMG0/grayscale/analyse');
        });

        it('should send a POST request', () => {
            API.SetGrayScaleImageAnalysis(...args);
            expect(mostRecentRequest().method)
                .toEqual('POST');
        });

        it('should send the access token', () => {
            API.SetGrayScaleImageAnalysis(...args);
            expect(mostRecentRequest().params.get('access_token'))
                .toEqual('T0K3N');
        });

        it('should return a promise that resolves on success', (done) => {
            API.SetGrayScaleImageAnalysis(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.SetGrayScaleImageAnalysis(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('SetGrayScaleTransform', () => {
        const args = ['CCC0', 'IMG0', 1, 'T0K3N'];

        it('should query the correct URL', () => {
            API.SetGrayScaleTransform(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/image/IMG0/plate/1/transform');
        });

        it('should send a POST request', () => {
            API.SetGrayScaleTransform(...args);
            expect(mostRecentRequest().method)
                .toEqual('POST');
        });

        it('should send the access_token', () => {
            API.SetGrayScaleTransform(...args);
            expect(mostRecentRequest().params.get('access_token'))
                .toEqual('T0K3N');
        });

        it('should return a promise that resolves on success', (done) => {
            API.SetGrayScaleTransform(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.SetGrayScaleTransform(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('GetFixturePlates', () => {
        const args = ['MyFixture']

        it('should query the correct URL', () => {
            API.GetFixturePlates(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/data/fixture/get/MyFixture');
        });

        it('should send a GET request', () => {
            API.GetFixturePlates(...args);
            expect(mostRecentRequest()).toHaveMethod('get');
        });

        it('should return a promise that resolves on success', (done) => {
            API.GetFixturePlates(...args).then(value => {
                expect(value).toEqual('xyz');
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ plates: 'xyz' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.GetFixturePlates(...args).catch(reason => {
                expect(reason).toEqual('bar');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: 'bar' }),
            });
        });
    });

    describe('SetNewCalibrationPolynomial', () => {
        const args = ['CCC0', '5' , 'T0K3N'];

        it('should query the correct URL', () => {
            API.SetNewCalibrationPolynomial(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/construct/5');
        });

        it('should send a POST request', () => {
            API.SetNewCalibrationPolynomial(...args);
            expect(mostRecentRequest().method)
                .toEqual('POST');
        });

        it('should send the access_token', () => {
            API.SetNewCalibrationPolynomial(...args);
            expect(JSON.parse(mostRecentRequest().params).access_token)
                .toEqual('T0K3N');
        });

        it('should return a promise that resolves on success', (done) => {
            API.SetNewCalibrationPolynomial(...args).then(value => {
                expect(value).toEqual({ foo: 'bar' });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ foo: 'bar' }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.SetNewCalibrationPolynomial(...args).catch(reason => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('GetFixture', () => {
        const args = [];

        it('should query the correct URL', () => {
            API.GetFixtures(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/data/fixture/names');
        });

        it('should send a GET request', () => {
            API.GetFixtures(...args);
            expect(mostRecentRequest()).toHaveMethod('get');
        });

        it('should return a promise that resolves on success', (done) => {
            API.GetFixtures(...args).then((value) => {
                expect(value).toEqual(['abc', 'xyz']);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({ fixtures: ['abc', 'xyz'] }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.GetFixtures(...args).catch((reason) => {
                expect(reason).toEqual('bar');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: 'bar' }),
            });
        });
    });

    describe('GetPinningFormats', () => {
        const args = [];

        it('should query the correct URL', () => {
            API.GetPinningFormats(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/analysis/pinning/formats');
        });

        it('should send a GET request', () => {
            API.GetPinningFormats(...args);
            expect(mostRecentRequest()).toHaveMethod('get');
        });

        it('should return a promise that resolves on success', (done) => {
            const apiData = {
                pinning_formats: [
                    { name: '1x1', value: [1, 1] },
                    { name: '2x4', value: [2, 4] },
                ],
            };
            const pinningFormats = [
                { name: '1x1', nCols: 1, nRows: 1 },
                { name: '2x4', nCols: 2, nRows: 4 },
            ];
            API.GetPinningFormats(...args).then((value) => {
                expect(value).toEqual(pinningFormats);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify(apiData),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.GetPinningFormats(...args).catch((reason) => {
                expect(reason).toEqual('bar');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: 'bar' }),
            });
        });
    });

    describe('InitiateCCC', () => {
        const species = 'S. Kombuchae';
        const reference = 'Professor X';
        const args = [species, reference];

        it('should query the correct URL', () => {
            API.InitiateCCC(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/initiate_new');
        });

        it('should send a POST request', () => {
            API.InitiateCCC(...args);
            expect(mostRecentRequest()).toHaveMethod('post');
        });

        it('should send the species', () => {
            API.InitiateCCC(...args);
            expect(mostRecentRequest().params.get('species')).toEqual(species);
        });

        it('should send the reference', () => {
            API.InitiateCCC(...args);
            expect(mostRecentRequest().params.get('reference')).toEqual(reference);
        });

        it('should return a promise that resolves on success', (done) => {
            const data = { id: 'CCC0', access_token: 'T0K3N' };
            API.InitiateCCC(...args).then((value) => {
                expect(value).toEqual(data);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify(data),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.InitiateCCC(...args).catch((reason) => {
                expect(reason).toEqual('bar');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: 'bar' }),
            });
        });
    });

    describe('finalizeCalibration', () => {
        const args = ['CCC0', 'T0K3N'];

        it('should query the correct URL', () => {
            API.finalizeCalibration(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/calibration/CCC0/finalize');
        });

        it('should send a POST request', () => {
            API.finalizeCalibration(...args);
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should send the access_token', () => {
            API.finalizeCalibration(...args);
            expect(JSON.parse(mostRecentRequest().params).access_token)
                .toEqual('T0K3N');
        });

        it('should return a promise that resolves on success', (done) => {
            API.finalizeCalibration(...args).then((value) => {
                expect(value).toEqual({});
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({}),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.finalizeCalibration(...args).catch((reason) => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });


    describe('getScanningJobs', () => {
        const jsonScanJob = {
            identifier: 'xyz',
            name: 'Some Job',
            scannerId: 'sc4nn3r',
            interval: 1200,
            duration: 97500,
            startTime: null,
        };

        it('should query the correct URL', () => {
            API.getScanningJobs();
            expect(mostRecentRequest().url)
                .toEqual('/api/scan-jobs');
        });

        it('should send a GET request', () => {
            API.getScanningJobs();
            expect(mostRecentRequest().method).toEqual('GET');
        });

        it('should return a promise that resolves on success', (done) => {
            API.getScanningJobs().then((value) => {
                expect(value).toEqual([jasmine.objectContaining({
                    identifier: 'xyz',
                    name: 'Some Job',
                    duration: new Duration(97500),
                    interval: new Duration(1200),
                    scannerId: 'sc4nn3r',
                })]);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify([jsonScanJob]),
            });
        });

        it('should parse startTime if present', (done) => {
            API.getScanningJobs().then((value) => {
                expect(value[0].startTime).toEqual(new Date('1985-10-26T01:20:00Z'));
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify([
                    Object.assign({}, jsonScanJob, {
                        startTime: '1985-10-26T01:20:00Z',
                    }),
                ]),
            });
        });

        it('should parse terminationTime if present', (done) => {
            API.getScanningJobs().then((value) => {
                expect(value[0].terminationTime)
                    .toEqual(new Date('1985-10-26T01:20:00Z'));
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify([
                    Object.assign({}, jsonScanJob, {
                        terminationTime: '1985-10-26T01:20:00Z',
                    }),
                ]),
            });
        });

        it('should set terminationMessage if present', (done) => {
            API.getScanningJobs().then((value) => {
                expect(value[0].terminationMessage).toEqual('The Message');
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,
                responseText: JSON.stringify([
                    Object.assign({}, jsonScanJob, {
                        terminationMessage: 'The Message',
                    }),
                ]),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.getScanningJobs().catch((reason) => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('startScanningJob', () => {
        const scanJob = {
            name: 'Some Job',
            duration: { days: 1, hours: 3, minutes: 5 },
            interval: 20,
            scannerId: 'sc4nn3r',
            identifier: '125dasd12',
        };

        it('should query the correct URL', () => {
            API.startScanningJob(scanJob);
            expect(mostRecentRequest().url)
                .toEqual('/api/scan-jobs/125dasd12/start');
        });

        it('should send a POST request', () => {
            API.startScanningJob(scanJob);
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should return a promise that resolves on success', (done) => {
            API.startScanningJob(scanJob).then((value) => {
                expect(value).toEqual(null);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify(null),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.getScanningJobs(scanJob).catch((reason) => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('submitScanningJob', () => {
        const scanJob = {
            name: 'Some Job',
            duration: { days: 1, hours: 3, minutes: 5 },
            interval: 20,
            scannerId: 'sc4nn3r',
        };
        const jsonScanJob = {
            name: 'Some Job',
            scannerId: 'sc4nn3r',
            interval: 1200,
            duration: 97500,
        };
        const args = [scanJob];

        it('should query the correct URL', () => {
            API.submitScanningJob(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/scan-jobs');
        });

        it('should send a POST request', () => {
            API.submitScanningJob(...args);
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should send the job', () => {
            API.submitScanningJob(...args);
            expect(JSON.parse(mostRecentRequest().params)).toEqual(jsonScanJob);
        });

        it('should return a promise that resolves on success', (done) => {
            API.submitScanningJob(...args).then((value) => {
                expect(value).toEqual({});
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify({}),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.submitScanningJob(...args).catch((reason) => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('getScannerJob', () => {
        const scanJob = {
            name: 'Some Job',
            duration: { days: 1, hours: 3, minutes: 5 },
            interval: 20,
            scannerId: 'sc4nn3r',
        };
        const args = ['sc4nn3r'];

        it('should query the correct URL', () => {
            API.getScannerJob(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/scanners/sc4nn3r/job');
        });

        it('should send a GET request', () => {
            API.getScannerJob(...args);
            expect(mostRecentRequest().method).toEqual('GET');
        });

        it('should return a promise that resolves on success', (done) => {
            API.getScannerJob(...args).then((value) => {
                expect(value).toEqual(scanJob);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify(scanJob),
            });
        });

        it('should return a promise that resolves if no job', (done) => {
            API.getScannerJob(...args).then((value) => {
                expect(value).toBe(null);
                done();
            });
            mostRecentRequest().respondWith({
                status: 200, responseText: 'null',
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.getScannerJob(...args).catch((reason) => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });

    describe('deleteScanningJob', () => {
        const args = ['scnjb005'];

        it('should query the correct URL', () => {
            API.deleteScanningJob(...args);
            expect(mostRecentRequest().url)
                .toEqual('/api/scan-jobs/scnjb005');
        });

        it('should send a DELETE request', () => {
            API.deleteScanningJob(...args);
            expect(mostRecentRequest().method).toEqual('DELETE');
        });

        it('should return a promise that resolves on success', (done) => {
            API.deleteScanningJob(...args).then(() => done());
            mostRecentRequest().respondWith({
                status: 200,
                responseText: 'null',
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.deleteScanningJob(...args).catch((reason) => {
                expect(reason).toEqual('No can do');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400,
                responseText: JSON.stringify({ reason: 'No can do' }),
            });
        });
    });

    describe('terminateScanningJob', () => {
        const scanJob = {
            name: 'Some Job',
            duration: { days: 1, hours: 3, minutes: 5 },
            interval: 20,
            scannerId: 'sc4nn3r',
            identifier: '125dasd12',
        };

        it('should query the correct URL', () => {
            API.terminateScanningJob('job002', 'Because fire!!!');
            expect(mostRecentRequest().url)
                .toEqual('/api/scan-jobs/job002/terminate');
        });

        it('should send a POST request', () => {
            API.terminateScanningJob('job002', 'Because fire!!!');
            expect(mostRecentRequest().method).toEqual('POST');
        });

        it('should send the message', () => {
            API.terminateScanningJob('job002', 'Because fire!!!');
            const params = JSON.parse(mostRecentRequest().params);
            expect(params.message).toEqual('Because fire!!!');
        });

        it('should return a promise that resolves on success', (done) => {
            API.terminateScanningJob('job002', 'Because fire!!!').then(done);
            mostRecentRequest().respondWith({
                status: 200, responseText: JSON.stringify(null),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.terminateScanningJob('job002', 'Because fire!!!').catch((reason) => {
                expect(reason).toEqual('(+_+)');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400, responseText: JSON.stringify({ reason: '(+_+)' }),
            });
        });
    });
});
