describe('SetGridding', ()=>{
    // function SetGridding(scope, cccId, imageId, plate, pinningFormat, offSet, accessToken, successCallback, errorCallback)

    const cccId = 'hello';
    const imageId = 'my-plate';
    const plate = 0;
    const pinningFormat = [42, 18];
    const offSet = [6, 7];
    const accessToken = 'open for me';

    beforeEach(() => {
        jasmine.Ajax.install();
    });

    afterEach(() => {
        jasmine.Ajax.uninstall();
    });

    it('Posts to the expected URI', () => {

        SetGridding(
            null,
            cccId,
            imageId,
            plate,
            pinningFormat,
            offSet,
            accessToken,
            null,
            null,
        );

        const uri = `${baseUrl}/api/calibration/${cccId}/image/${imageId}/plate/${plate}/grid/set`;
        expect(jasmine.Ajax.requests.mostRecent().url).toBe(uri);
    });

    it('calls successCallback with expected arguments', ()=>{
        const responseJSON = {hello: 'world'};
        const responseText = JSON.stringify(responseJSON);
        const successCallback = jasmine.createSpy('success');
        const errorCallback = jasmine.createSpy('error');
        const scope = {1: 2, 3: 4};

        SetGridding(
            scope,
            cccId,
            imageId,
            plate,
            pinningFormat,
            offSet,
            accessToken,
            successCallback,
            errorCallback,
        );

        jasmine.Ajax.requests.mostRecent().respondWith({
            "status": 200,
            responseText
        });

        expect(successCallback).toHaveBeenCalledWith(responseJSON, scope);
        expect(errorCallback).not.toHaveBeenCalled();
    });

    it('calls errorCallback with expected arguments', ()=>{
        const responseJSON = {hello: 'world'};
        const responseText = JSON.stringify(responseJSON);
        const successCallback = jasmine.createSpy('success');
        const errorCallback = jasmine.createSpy('error');
        const scope = {1: 2, 3: 4};

        SetGridding(
            scope,
            cccId,
            imageId,
            plate,
            pinningFormat,
            offSet,
            accessToken,
            successCallback,
            errorCallback,
        );

        jasmine.Ajax.requests.mostRecent().respondWith({
            "status": 400,
            responseText
        });

        expect(errorCallback).toHaveBeenCalledWith(responseJSON, scope);
        expect(successCallback).not.toHaveBeenCalled();
    });
});
