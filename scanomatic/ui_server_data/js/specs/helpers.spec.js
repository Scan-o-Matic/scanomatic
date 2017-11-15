import { loadImage, uploadImage } from '../ccc/helpers';
import testPlateImageURL from './fixtures/testPlate.png';
import * as API from '../ccc/api';

describe('loadImage()', () => {
    it('should return a promise that resolves for a valid image URL', (done) => {
        loadImage(testPlateImageURL).then(() => done());
    });

    it('should return a promise that rejects for a bad image URL', (done) => {
        loadImage('/this/is/not/a/valid/image.png').catch(() => done());
    });
});

describe('UploadImage', () => {
    const cccId = 'CCC0';
    const imageId = 'IMG0';
    const fixture = 'MyFixture';
    const file = new File(['foo'], 'myimage.tiff');
    const token = 'T0K3N1';
    const markers = [[1, 2], [3, 4], [5, 6]];

    const args = [cccId, file, fixture, token];

    beforeEach(() => {
        spyOn(API, 'GetMarkers')
            .and.callFake(() => Promise.resolve({ markers }));
        spyOn(API, 'GetImageId')
            .and.callFake(() => Promise.resolve({ image_identifier: imageId }));
        spyOn(API, 'SetCccImageData')
            .and.callFake(() => Promise.resolve({}));
        spyOn(API, 'SetCccImageSlice')
            .and.callFake(() => Promise.resolve({}));
        spyOn(API, 'SetGrayScaleImageAnalysis')
            .and.callFake(() => Promise.resolve({}));
    });

    it('should call GetMarkers', (done) => {
        uploadImage(...args).then(() => {
            expect(API.GetMarkers).toHaveBeenCalledWith(fixture, file);
            done();
        });
    });

    it('should reject if GetMarkers rejects', (done) => {
        API.GetMarkers
            .and.callFake(() => Promise.reject('Whoopsie'));
        uploadImage(...args).catch(reason => {
            expect(reason).toEqual('Whoopsie');
            done();
        });
    });

    it('should call GetImageId', (done) => {
        uploadImage(...args).then(() => {
            expect(API.GetImageId).toHaveBeenCalledWith(cccId, file, token);
            done()
        });
    });

    it('should reject if GetImageId rejects', (done) => {
        API.GetImageId
            .and.callFake(() => Promise.reject('Whoopsie'));
        uploadImage(...args).catch(reason => {
            expect(reason).toEqual('Whoopsie');
            done();
        });
    });

    it('should call SetCccImageData', (done) => {
        const data = [
            { key: "marker_x", value: [1, 3, 5] },
            { key: "marker_y", value: [2, 4, 6] }
        ];
        uploadImage(...args).then(() => {
            expect(API.SetCccImageData)
                .toHaveBeenCalledWith(cccId, imageId, token, data, fixture);
            done();
        });
    });

    it('should reject if SetCccImageData rejects', (done) => {
        API.SetCccImageData
            .and.callFake(() => Promise.reject('Whoopsie'));
        uploadImage(...args).catch(reason => {
            expect(reason).toEqual('Whoopsie');
            done();
        });
    });

    it('should call SetCccImageSlice', (done) => {
        uploadImage(...args).then(() => {
            expect(API.SetCccImageSlice)
                .toHaveBeenCalledWith(cccId, imageId, token);
            done();
        });
    });

    it('should reject if SetCccImageSlice rejects', (done) => {
        API.SetCccImageSlice
            .and.callFake(() => Promise.reject('Whoopsie'));
        uploadImage(...args).catch(reason => {
            expect(reason).toEqual('Whoopsie');
            done();
        });
    });

    it('should call SetGrayScaleImageAnalysis', (done) => {
        uploadImage(...args).then(() => {
            expect(API.SetGrayScaleImageAnalysis)
                .toHaveBeenCalledWith(cccId, imageId, token);
            done();
        });
    });

    it('should reject if SetGrayScaleImageAnalysis rejects', (done) => {
        API.SetGrayScaleImageAnalysis
            .and.callFake(() => Promise.reject('Whoopsie'));
        uploadImage(...args).catch(reason => {
            expect(reason).toEqual('Whoopsie');
            done();
        });
    });

    it('should return a promise with the image id', (done) => {
        uploadImage(...args).then(value => {
            expect(value).toEqual(imageId);
            done();
        });
    });
});
