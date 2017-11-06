import { loadImage } from '../ccc/helpers';
import testPlateImageURL from './fixtures/testPlate.png';

describe('loadImage()', () => {
    it('should return a promise that resolves for a valid image URL', (done) => {
        loadImage(testPlateImageURL).then(() => done());
    });

    it('should return a promise that rejects for a bad image URL', (done) => {
        loadImage('/this/is/not/a/valid/image.png').catch(() => done());
    });
});
