
function sliceImageUp(x1,y1,x2,y2, image) {
    var canvas = document.createElement('canvas');
    var sliceWidth = x2 - x1;
    var sliceHeight = y2 - y1;

    canvas.width = sliceWidth;
    canvas.height = sliceHeight;
    var context = canvas.getContext('2d');
    context.drawImage(image, x1, y1, sliceWidth, sliceHeight, 0, 0, canvas.width, canvas.height);
    var slicedImage = canvas.toDataURL();
    return slicedImage;
}

function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], { type: mimeString });
}