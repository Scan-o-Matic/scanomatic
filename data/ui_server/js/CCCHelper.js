﻿
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

function getDataUrlfromUrl(src, callback) {
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


function createCanvasImage(data, canvas) {

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

function createCanvasMarker(data, canvas) {
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

function getMarkerData(canvasId) {

    var cvPlot = document.getElementById(canvasId);
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

function hexToRgb(hex) {
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

function getLinearMapping(data) {

    var colorScheme = ["white", "grey", "black"];
    var intensityMin = data.blobMin;
    var intensityMax = data.imageMax;
    var intensityMean = (intensityMax + intensityMin) / 2;

    var cs = d3.scale.linear()
        .domain([intensityMin, intensityMean, intensityMax])
        .range([colorScheme[2], colorScheme[1], colorScheme[0]]);

    return cs;
}



