import {
    RGBColor,
    getScannersWithOwned,
    loadImage,
    uploadImage,
    valueFormatter,
} from '../src/helpers';
import testPlateImageURL from './fixtures/testPlate.png';
import * as API from '../src/api';

describe('helpers', () => {
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
        const progress = jasmine.createSpy('progress');

        const args = [cccId, file, fixture, token, progress];

        beforeEach(() => {
            progress.calls.reset();
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

        it('should set progress to 0/5 "Getting markers"', (done) => {
            uploadImage(...args).then(() => {
                expect(progress).toHaveBeenCalledWith(0, 5, 'Getting markers');
                done();
            });
        });

        it('should reject if GetMarkers rejects', (done) => {
            API.GetMarkers
                .and.callFake(() => Promise.reject('Whoopsie'));
            uploadImage(...args).catch((reason) => {
                expect(reason).toEqual('Whoopsie');
                done();
            });
        });

        it('should call GetImageId', (done) => {
            uploadImage(...args).then(() => {
                expect(API.GetImageId).toHaveBeenCalledWith(cccId, file, token);
                done();
            });
        });

        it('should set progress to 1/5 "Uploading image"', (done) => {
            uploadImage(...args).then(() => {
                expect(progress).toHaveBeenCalledWith(1, 5, 'Uploading image');
                done();
            });
        });

        it('should reject if GetImageId rejects', (done) => {
            API.GetImageId
                .and.callFake(() => Promise.reject('Whoopsie'));
            uploadImage(...args).catch((reason) => {
                expect(reason).toEqual('Whoopsie');
                done();
            });
        });

        it('should call SetCccImageData', (done) => {
            const data = [
                { key: 'marker_x', value: [1, 3, 5] },
                { key: 'marker_y', value: [2, 4, 6] },
            ];
            uploadImage(...args).then(() => {
                expect(API.SetCccImageData)
                    .toHaveBeenCalledWith(cccId, imageId, token, data, fixture);
                done();
            });
        });

        it('should set progress to 2/5 "Setting image CCC data"', (done) => {
            uploadImage(...args).then(() => {
                expect(progress).toHaveBeenCalledWith(2, 5, 'Setting image CCC data');
                done();
            });
        });

        it('should reject if SetCccImageData rejects', (done) => {
            API.SetCccImageData
                .and.callFake(() => Promise.reject('Whoopsie'));
            uploadImage(...args).catch((reason) => {
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

        it('should set progress to 3/5 "Slicing image"', (done) => {
            uploadImage(...args).then(() => {
                expect(progress).toHaveBeenCalledWith(3, 5, 'Slicing image');
                done();
            });
        });

        it('should reject if SetCccImageSlice rejects', (done) => {
            API.SetCccImageSlice
                .and.callFake(() => Promise.reject('Whoopsie'));
            uploadImage(...args).catch((reason) => {
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

        it('should set progress to 4/5 "Setting grayscase"', (done) => {
            uploadImage(...args).then(() => {
                expect(progress).toHaveBeenCalledWith(4, 5, 'Setting grayscale');
                done();
            });
        });

        it('should reject if SetGrayScaleImageAnalysis rejects', (done) => {
            API.SetGrayScaleImageAnalysis
                .and.callFake(() => Promise.reject('Whoopsie'));
            uploadImage(...args).catch((reason) => {
                expect(reason).toEqual('Whoopsie');
                done();
            });
        });

        it('should return a promise with the image id', (done) => {
            uploadImage(...args).then((value) => {
                expect(value).toEqual(imageId);
                done();
            });
        });
    });

    describe('RGBColor', () => {
        it('should have attribute r, g and b', () => {
            const color = new RGBColor(0, 128, 255);
            expect(color.r).toEqual(0);
            expect(color.g).toEqual(128);
            expect(color.b).toEqual(255);
        });

        it('should be able to generate a CSS string representation of the color', () => {
            const color = new RGBColor(0, 128, 255);
            expect(color.toCSSString()).toEqual('rgb(0, 128, 255)');
        });
    });

    describe('labelFormatter', () => {
        it('returns zero', () => {
            expect(valueFormatter(0)).toEqual('0');
        });

        it('returns the expected output for value', () => {
            expect(valueFormatter(320)).toEqual('3 x 10^2');
        });

        it('respects fixed positions', () => {
            expect(valueFormatter(320, 1)).toEqual('3.2 x 10^2');
        });

        it('works with negative numbers', () => {
            expect(valueFormatter(-320, 1)).toEqual('-3.2 x 10^2');
        });
    });

    describe('getScannersWithOwned', () => {
        const scanners = [
            {
                identifier: 'sc4nn3r01',
                name: 'Scanner 01',
                owned: false,
                power: true,
            },
            {
                identifier: 'sc4nn3r02',
                name: 'Scanner 02',
                owned: true,
                power: true,
            },
        ];

        const apiScanners = scanners.map(obj => ({
            identifier: obj.identifier,
            name: obj.name,
            power: obj.power,
        }));

        it('should return a list of scanners on success', (done) => {
            spyOn(API, 'getScanners')
                .and.callFake(() => Promise.resolve(apiScanners));
            spyOn(API, 'getScannerJob')
                .and.callFake((id) => {
                    const job = id === 'sc4nn3r02' ? { identifier: 'job0123' } : null;
                    return Promise.resolve(job);
                });
            getScannersWithOwned().then((data) => {
                expect(data).toEqual(scanners);
                done();
            });
        });

        it('should reject if getScanners rejects', (done) => {
            const message = 'Sorry, Dave.';
            spyOn(API, 'getScanners')
                .and.callFake(() => Promise.reject(message));
            getScannersWithOwned().catch((error) => {
                expect(error).toEqual(message);
                done();
            });
        });

        it('should reject if getScannerJob rejects', (done) => {
            const message = "I'm afraid I can't do that.";
            spyOn(API, 'getScanners')
                .and.callFake(() => Promise.resolve(apiScanners));
            spyOn(API, 'getScannerJob')
                .and.callFake(() => Promise.reject(message));
            getScannersWithOwned().catch((error) => {
                expect(error).toEqual(message);
                done();
            });
        });
    });
});
