import * as API from './api';


export class RGBColor {
    constructor(r, g, b) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    toCSSString() {
        return `rgb(${this.r}, ${this.g}, ${this.b})`;
    }
}


export const featureColors = {
    blob: new RGBColor(253, 231, 35),
    background: new RGBColor(32, 144, 140),
    neither: new RGBColor(68, 1, 84),
};


export const valueFormatter = (value, fixed = 0) => {
    if (value === 0) {
        return value.toFixed(0);
    }
    const exponent = Math.floor(Math.log10(Math.abs(value)));
    const number = (value / (10 ** exponent)).toFixed(fixed);
    return `${number} x 10^${exponent.toFixed(0)}`;
};


export function getDataUrlfromUrl(src, callback) {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
        const canvas = document.createElement('CANVAS');
        const ctx = canvas.getContext('2d');
        let dataURL;
        canvas.height = this.height;
        canvas.width = this.width;
        ctx.drawImage(this, 0, 0);
        dataURL = canvas.toDataURL();
        callback(dataURL);
    };
    img.src = src;
    if (img.complete || img.complete === undefined) {
        img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
        img.src = src;
    }
}


export function createCanvasImage(data, canvas) {
    const cs = getLinearMapping(data);
    const rows = data.image.length;
    const cols = data.image[0].length;
    const cvPlot = canvas;
    cvPlot.width = cols;
    cvPlot.height = rows;
    const ctx = cvPlot.getContext('2d');
    const imgdata = ctx.getImageData(0, 0, cols, rows);
    const imgdatalen = imgdata.data.length;
    let imageCol = -1;
    let imageRow = 0;
    for (let i = 0; i < imgdatalen; i += 4) { // iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        const color = data.image[imageRow][imageCol];
        const mappedColor = cs(color);

        const rgb = hexToRgb(mappedColor);

        imgdata.data[i + 0] = rgb.r; // RED (0-255)
        imgdata.data[i + 1] = rgb.g; // GREEN (0-255)
        imgdata.data[i + 2] = rgb.b; // BLUE (0-255)
        imgdata.data[i + 3] = 255; // APLHA (0-255)
    }
    ctx.putImageData(imgdata, 0, 0);
}

export function createCanvasMarker(data, canvas) {
    const rows = data.image.length;
    const cols = data.image[0].length;
    const cvPlot = canvas;
    cvPlot.width = cols;
    cvPlot.height = rows;
    const ctx = cvPlot.getContext('2d');
    const imgdata = ctx.getImageData(0, 0, cols, rows);
    const imgdatalen = imgdata.data.length;
    let imageCol = -1;
    let imageRow = 0;
    for (let i = 0; i < imgdatalen; i += 4) { // iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        var rgb;
        if (data.blob[imageRow][imageCol] === true) {
            rgb = featureColors.blob;
        } else if (data.background[imageRow][imageCol] === true) {
            rgb = featureColors.background;
        } else {
            rgb = featureColors.neither;
        }

        imgdata.data[i + 0] = rgb.r; // RED (0-255)
        imgdata.data[i + 1] = rgb.g; // GREEN (0-255)
        imgdata.data[i + 2] = rgb.b; // BLUE (0-255)
        imgdata.data[i + 3] = 255; // APLHA (0-255)
    }
    ctx.putImageData(imgdata, 0, 0);
}

export function getMarkerData(cvPlot) {
    const rows = cvPlot.height;
    const cols = cvPlot.width;

    const ctx = cvPlot.getContext('2d');
    const imgdata = ctx.getImageData(0, 0, cols, rows);
    const imgdatalen = imgdata.data.length;
    let imageCol = -1;
    let imageRow = 0;

    const blob = new Array(rows);
    for (var i = 0; i < rows; i++) {
        blob[i] = new Array(cols);
    }

    const background = new Array(rows);
    for (var i = 0; i < rows; i++) {
        background[i] = new Array(cols);
    }

    for (var i = 0; i < imgdatalen; i += 4) { // iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        const r = imgdata.data[i + 0];
        const g = imgdata.data[i + 1];
        const b = imgdata.data[i + 2];
        const a = imgdata.data[i + 3];
        const pixelRgbA = `${r},${g},${b}`;

        if (pixelRgbA === '255,0,0') {
            blob[imageRow][imageCol] = true;
            background[imageRow][imageCol] = false;
        } else if (pixelRgbA === '0,0,0') {
            blob[imageRow][imageCol] = false;
            background[imageRow][imageCol] = true;
        }
    }
    const x = { blob, background };
    return x;
}

export function hexToRgb(hex) {
    // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, (m, r, g, b) => r + r + g + g + b + b);

    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
    } : null;
}

export function getLinearMapping(data) {
    const colorScheme = ['white', 'grey', 'black'];
    const intensityMin = data.imageMin;
    const intensityMax = data.imageMax;
    const intensityMean = (intensityMax + intensityMin) / 2;

    const cs = d3.scale.linear()
        .domain([intensityMin, intensityMean, intensityMax])
        .range([colorScheme[2], colorScheme[1], colorScheme[0]]);

    return cs;
}

export function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => {
            console.log(`Could not load ${url}`);
            return reject();
        };
        image.src = url;
    });
}

export function uploadImage(ccc, file, fixture, token, progress) {
    const markers = [];
    let imageId;
    progress(0, 5, 'Getting markers');
    return API.GetMarkers(fixture, file).then((data) => {
        markers[0] = data.markers.map(xy => xy[0]);
        markers[1] = data.markers.map(xy => xy[1]);
        progress(1, 5, 'Uploading image');
        return API.GetImageId(ccc, file, token);
    }).then((data) => {
        imageId = data.image_identifier;
        const imageData = [
            { key: 'marker_x', value: markers[0] },
            { key: 'marker_y', value: markers[1] },
        ];
        progress(2, 5, 'Setting image CCC data');
        return API.SetCccImageData(ccc, imageId, token, imageData, fixture);
    }).then(() => {
        progress(3, 5, 'Slicing image');
        return API.SetCccImageSlice(ccc, imageId, token);
    })
        .then(() => {
            progress(4, 5, 'Setting grayscale');
            return API.SetGrayScaleImageAnalysis(ccc, imageId, token);
        })
        .then(() => imageId);
}
