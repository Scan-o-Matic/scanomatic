
export default function toLookLikeImage(util, customEqualityTesters) {
    return {
        compare: (actual, expected) => {
            const expectedCanvas = document.createElement('canvas');
            expectedCanvas.width = expected.naturalWidth;
            expectedCanvas.height = expected.naturalHeight;
            expectedCanvas.getContext('2d').drawImage(expected, 0, 0);

            const actualImageData = actual.getContext('2d')
                .getImageData(0, 0, actual.width, actual.height);
            const expectedImageData = expectedCanvas.getContext('2d')
                .getImageData(0, 0, actual.width, actual.height);
            const result = {};
            result.pass = util.equals(actualImageData, expectedImageData, customEqualityTesters);

            const actualDataURL = actual.toDataURL();
            const expectedDataURL = expectedCanvas.toDataURL();
            if (result.pass) {
                result.message = 'Expected canvas to look different';
            } else {
                result.message = 'Expected canvas to look the same';
            }
            result.message += `\n\texpected: ${expectedDataURL}`;
            result.message += `\n\tactual: ${actualDataURL}`;
            return result;
        },
    };
}
