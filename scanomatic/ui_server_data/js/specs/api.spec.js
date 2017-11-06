import { SetColonyCompression, SetColonyDetection, SetGridding } from '../ccc/api';

describe('API', () => {
    describe('SetGridding', () => {
        const cccId = 'hello';
        const imageId = 'my-plate';
        const plate = 0;
        const pinningFormat = [42, 18];
        const offSet = [6, 7];
        const accessToken = 'open for me';
        const successCallback = jasmine.createSpy('success');
        const errorCallback = jasmine.createSpy('error');
        const args = [
            cccId, imageId, plate, pinningFormat, offSet, accessToken,
            successCallback, errorCallback,
        ];

        beforeEach(() => {
            successCallback.calls.reset();
            errorCallback.calls.reset();
            jasmine.Ajax.install();
        });

        afterEach(() => {
            jasmine.Ajax.uninstall();
        });

        it('Posts to the expected URI', () => {

            SetGridding(...args);

            const uri = `/api/calibration/${cccId}/image/${imageId}/plate/${plate}/grid/set`;
            expect(jasmine.Ajax.requests.mostRecent().url).toBe(uri);
        });

        it('calls successCallback with expected arguments', ()=>{
            const responseJSON = {hello: 'world'};
            const responseText = JSON.stringify(responseJSON);

            SetGridding(...args);

            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 200,
                responseText
            });
            expect(successCallback).toHaveBeenCalledWith(responseJSON);
            expect(errorCallback).not.toHaveBeenCalled();
        });

        it('calls errorCallback with expected arguments', ()=>{
            const responseJSON = {hello: 'world'};
            const responseText = JSON.stringify(responseJSON);

            SetGridding(...args);

            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 400,
                responseText
            });

            expect(errorCallback).toHaveBeenCalledWith(responseJSON);
            expect(successCallback).not.toHaveBeenCalled();
        });
    });

    describe('SetColonyCompression', () => {
        const onSuccess = jasmine.createSpy('onSuccess');
        const onError = jasmine.createSpy('onError');
        const args = [
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
            onSuccess,
            onError,
        ];

        beforeEach(() => {
            onSuccess.calls.reset();
            onError.calls.reset();
            jasmine.Ajax.install();
        });

        afterEach(() => {
            jasmine.Ajax.uninstall();
        });

        it('should query the correct url', () => {
            SetColonyCompression(...args);
            expect(jasmine.Ajax.requests.mostRecent().url)
                .toBe('/api/data/calibration/CCC42/image/1M4G3/plate/PL4T3/compress/colony/1/4');
        });

        it('should call onSuccess on success', () => {
            const data = { foo: 'bar' };
            SetColonyCompression(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 200, responseText: JSON.stringify(data),
            });
            expect(onSuccess).toHaveBeenCalledWith(data);
        });

        it('should call onError on error', () => {
            const data = { foo: 'bar' };
            SetColonyCompression(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 400, responseText: JSON.stringify(data),
            });
            expect(onError).toHaveBeenCalledWith(data);
        });
    })

    describe('SetColonyDetection', () => {
        const onSuccess = jasmine.createSpy('onSuccess');
        const onError = jasmine.createSpy('onError');
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
            jasmine.Ajax.install();
        });

        afterEach(() => {
            jasmine.Ajax.uninstall();
        });

        it('should query the correct url', () => {
            SetColonyDetection(...args);
            expect(jasmine.Ajax.requests.mostRecent().url)
                .toBe('/api/data/calibration/CCC42/image/1M4G3/plate/PL4T3/detect/colony/1/4');
        });

        it('should call onSuccess on success', () => {
            const data = { foo: 'bar' };
            SetColonyDetection(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 200, responseText: JSON.stringify(data),
            });
            expect(onSuccess).toHaveBeenCalledWith(data);
        });

        it('should call onError on error', () => {
            const data = { foo: 'bar' };
            SetColonyDetection(...args);
            jasmine.Ajax.requests.mostRecent().respondWith({
                status: 400, responseText: JSON.stringify(data),
            });
            expect(onError).toHaveBeenCalledWith(data);
        });
    })
});
