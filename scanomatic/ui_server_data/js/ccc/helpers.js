import * as API from './api';


export function getDataUrlfromUrl(src, callback) {
    var img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
        var canvas = document.createElement('CANVAS');
        var ctx = canvas.getContext('2d');
        var dataURL;
        canvas.height = this.height;
        canvas.width = this.width;
        ctx.drawImage(this, 0, 0);
        dataURL = canvas.toDataURL();
        callback(dataURL);
    };
    img.src = src;
    if (img.complete || img.complete === undefined) {
        img.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
        img.src = src;
    }
}


export function createCanvasImage(data, canvas) {
    var cs = getLinearMapping(data);
    var rows = data.image.length;
    var cols = data.image[0].length;
    var cvPlot = canvas;
    cvPlot.width = cols;
    cvPlot.height = rows;
    var ctx = cvPlot.getContext('2d');
    var imgdata = ctx.getImageData(0, 0, cols, rows);
    var imgdatalen = imgdata.data.length;
    var imageCol = -1;
    var imageRow = 0;
    for (var i = 0; i < imgdatalen; i += 4) {  //iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        var color = data.image[imageRow][imageCol];
        var mappedColor = cs(color);

        var rgb = hexToRgb(mappedColor);

        imgdata.data[i + 0] = rgb.r;    // RED (0-255)
        imgdata.data[i + 1] = rgb.g;    // GREEN (0-255)
        imgdata.data[i + 2] = rgb.b;    // BLUE (0-255)
        imgdata.data[i + 3] = 255;  // APLHA (0-255)

    }
    ctx.putImageData(imgdata, 0, 0);
}

export function createCanvasMarker(data, canvas) {
    var rows = data.image.length;
    var cols = data.image[0].length;
    var cvPlot = canvas;
    cvPlot.width = cols;
    cvPlot.height = rows;
    var ctx = cvPlot.getContext('2d');
    var imgdata = ctx.getImageData(0, 0, cols, rows);
    var imgdatalen = imgdata.data.length;
    var imageCol = -1;
    var imageRow = 0;
    for (var i = 0; i < imgdatalen; i += 4) {  //iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        var rgb;
        if (data.blob[imageRow][imageCol] === true) {
            rgb = { r: 255, g: 0, b: 0 }
        } else if (data.background[imageRow][imageCol] === true) {
            rgb = { r: 0, g: 128, b: 0 }
        } else {
            rgb = { r: 0, g: 0, b: 0 };
        }

        imgdata.data[i + 0] = rgb.r;    // RED (0-255)
        imgdata.data[i + 1] = rgb.g;    // GREEN (0-255)
        imgdata.data[i + 2] = rgb.b;    // BLUE (0-255)
        imgdata.data[i + 3] = 255;  // APLHA (0-255)
    }
    ctx.putImageData(imgdata, 0, 0);
}

export function getMarkerData(cvPlot) {

    var rows = cvPlot.height;
    var cols = cvPlot.width;

    var ctx = cvPlot.getContext('2d');
    var imgdata = ctx.getImageData(0, 0, cols, rows);
    var imgdatalen = imgdata.data.length;
    var imageCol = -1;
    var imageRow = 0;

    var blob = new Array(rows);
    for (var i = 0; i < rows; i++) {
        blob[i] = new Array(cols);
    }

    var background = new Array(rows);
    for (var i = 0; i < rows; i++) {
        background[i] = new Array(cols);
    }

    for (var i = 0; i < imgdatalen; i += 4) {  //iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        var r = imgdata.data[i + 0];
        var g = imgdata.data[i + 1];
        var b = imgdata.data[i + 2];
        var a = imgdata.data[i + 3];
        var pixelRgbA = r + "," + g + "," + b;

        if (pixelRgbA === "255,0,0") {
            blob[imageRow][imageCol] = true;
            background[imageRow][imageCol] = false;
        } else if (pixelRgbA === "0,0,0") {
            blob[imageRow][imageCol] = false;
            background[imageRow][imageCol] = true;
        };
    }
    var x = { blob: blob, background: background };
    return x;
}

export function hexToRgb(hex) {
    // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
    var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, function (m, r, g, b) {
        return r + r + g + g + b + b;
    });

    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

export function getLinearMapping(data) {

    var colorScheme = ["white", "grey", "black"];
    var intensityMin = data.imageMin;
    var intensityMax = data.imageMax;
    var intensityMean = (intensityMax + intensityMin) / 2;

    var cs = d3.scale.linear()
        .domain([intensityMin, intensityMean, intensityMax])
        .range([colorScheme[2], colorScheme[1], colorScheme[0]]);

    return cs;
}

export function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image;
        image.onload = () => resolve(image);
        image.onerror = () => reject();
        image.src = url;
    });
}

export function uploadImage(ccc, file, fixture, token) {
    const markers = [];
    let imageId;
    return API.GetMarkers(fixture, file).then(data => {
        markers[0] = data.markers.map(xy => xy[0]);
        markers[1] = data.markers.map(xy => xy[1]);
        return API.GetImageId(ccc, file, token);
    }).then(data => {
        imageId = data.image_identifier;
        const imageData = [
            { key: 'marker_x', value: markers[0] },
            { key: 'marker_y', value: markers[1] },
        ];
        return API.SetCccImageData(ccc, imageId, token, imageData, fixture);
    }).then(() => {
        return API.SetCccImageSlice(ccc, imageId, token);
    }).then(() => {
        return API.SetGrayScaleImageAnalysis(ccc, imageId, token);
    }).then(() => {
        return imageId;
    });
}
