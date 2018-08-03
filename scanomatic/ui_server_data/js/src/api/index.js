import $ from 'jquery';
import Duration from '../Duration';
import API from './API';

import _getPlateGrowthData from './getPlateGrowthData';
import _getPhentypeData from './getPhenotypeData';
import { setCurveQCMark as _setCurveQCMark, setCurveQCMarkAll as _setCurveQCMarkAll } from './setCurveQCMark';

export const getPlateGrowthData = _getPlateGrowthData;
export const getPhenotypeData = _getPhentypeData;
export const setCurveQCMark = _setCurveQCMark;
export const setCurveQCMarkAll = _setCurveQCMarkAll;

const GetSliceImagePath = '/api/calibration/#0#/image/#1#/slice/get/#2#';
const GetTranposedMarkerPath = '/api/data/fixture/calculate/';
const GetGrayScaleAnalysisPath = '/api/data/grayscale/image/';

const secondsPerMinute = 60;
const secondsPerHour = 3600;
const secondsPerDay = 86400;

export function GetSliceImageURL(cccId, imageId, slice) {
    const path = GetSliceImagePath.replace('#0#', cccId).replace('#1#', imageId).replace('#2#', slice);
    return path;
}

export function GetSliceImage(cccId, imageId, slice, successCallback, errorCallback) {
    const path = GetSliceImagePath.replace('#0#', cccId).replace('#1#', imageId).replace('#2#', slice);

    $.get(path, successCallback).fail(errorCallback);
}


export function GetFixtures() {
    return API.get('/api/data/fixture/names').then(data => data.fixtures);
}

function GetFixtureData(fixtureName) {
    const path = `/api/data/fixture/get/${fixtureName}`;
    return API.get(path);
}

export function GetFixturePlates(fixtureName) {
    return GetFixtureData(fixtureName).then(data => data.plates);
}

export function GetPinningFormats() {
    return API.get('/api/analysis/pinning/formats')
        .then(data => data.pinning_formats.map(({ name, value }) => (
            { name, nCols: value[0], nRows: value[1] }
        )));
}

export function InitiateCCC(species, reference, successCallback, errorCallback) {
    const formData = new FormData();
    formData.append('species', species);
    formData.append('reference', reference);
    return API.postFormData('/api/calibration/initiate_new', formData);
}

export function SetCccImageData(cccId, imageId, accessToken, dataArray, fixture) {
    const path = `/api/calibration/${cccId}/image/${imageId}/data/set`;
    const formData = new FormData();
    formData.append('ccc_identifier', cccId);
    formData.append('image_identifier', imageId);
    formData.append('access_token', accessToken);
    formData.append('fixture', fixture);
    for (let i = 0; i < dataArray.length; i++) {
        const item = dataArray[i];
        formData.append(item.key, item.value);
    }
    return API.postFormData(path, formData);
}

export function SetCccImageSlice(cccId, imageId, accessToken) {
    const path = `/api/calibration/${cccId}/image/${imageId}/slice/set`;
    const formData = new FormData();
    formData.append('access_token', accessToken);
    return API.postFormData(path, formData);
}

export function SetGrayScaleImageAnalysis(cccId, imageId, accessToken) {
    const path = `/api/calibration/${cccId}/image/${imageId}/grayscale/analyse`;
    const formData = new FormData();
    formData.append('access_token', accessToken);
    return API.postFormData(path, formData);
}

export function SetGrayScaleTransform(cccId, imageId, plate, accessToken) {
    const path = `/api/calibration/${cccId}/image/${imageId}/plate/${plate}/transform`;
    const formData = new FormData();
    formData.append('access_token', accessToken);
    return API.postFormData(path, formData);
}

export function SetGridding(cccId, imageId, plate, pinningFormat, offSet, accessToken) {
    const path = `/api/calibration/${cccId}/image/${imageId}/plate/${plate}/grid/set`;
    return new Promise((resolve, reject) => $.ajax({
        url: path,
        type: 'POST',
        dataType: 'json',
        contentType: 'application/json',
        data: JSON.stringify({
            pinning_format: pinningFormat,
            gridding_correction: offSet,
            access_token: accessToken,
        }),
        success: resolve,
        error: jqXHR => reject(JSON.parse(jqXHR.responseText)),
    }));
}

export function SetColonyDetection(cccId, imageId, plate, accessToken, row, col, successCallback, errorCallback) {
    const path = `/api/calibration/${cccId}/image/${imageId}/plate/${plate}/detect/colony/${col}/${row}`;

    const formData = new FormData();
    formData.append('access_token', accessToken);
    $.ajax({
        url: path,
        type: 'POST',
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: data => successCallback(data),
        error: jqXHR => errorCallback(JSON.parse(jqXHR.responseText)),
    });
}

export function SetColonyCompression(cccId, imageId, plate, accessToken, colony, cellCount, row, col, successCallback, errorCallback) {
    const path = `/api/calibration/${cccId}/image/${imageId}/plate/${plate}/compress/colony/${col}/${row}`;

    const data = {
        access_token: accessToken,
        image: colony.image,
        blob: colony.blob,
        background: colony.background,
        cell_count: cellCount,
    };
    $.ajax({
        url: path,
        method: 'POST',
        data: JSON.stringify(data),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',
        success(data) {
            successCallback(data);
        },
        error: jqXHR => errorCallback(JSON.parse(jqXHR.responseText)),
    });
}

export function GetImageId(cccId, file, accessToken) {
    const path = `/api/calibration/${cccId}/add_image`;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('access_token', accessToken);
    return API.postFormData(path, formData);
}

export function GetMarkers(fixtureName, file) {
    const path = `/api/data/markers/detect/${fixtureName}`;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('save', 'false');
    return API.postFormData(path, formData);
}

export function GetTransposedMarkersV2(fixtureName, markers, file, successCallback, errorCallback) {
    const path = GetTranposedMarkerPath + fixtureName;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('markers', markers);
    $.ajax({
        url: path,
        type: 'POST',
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback,
    });
}

export function GetTransposedMarkers(fixtureName, markers, successCallback, errorCallback) {
    const path = GetTranposedMarkerPath + fixtureName;
    const formData = new FormData();
    formData.append('markers', markers);
    $.ajax({
        url: path,
        type: 'POST',
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback,
    });
}

export function SetNewCalibrationPolynomial(cccId, power, accessToken) {
    return API.postJSON(
        `/api/calibration/${cccId}/construct/${power}`,
        { access_token: accessToken },
    );
}

export function finalizeCalibration(cccId, accessToken) {
    return API.postJSON(
        `/api/calibration/${cccId}/finalize`,
        { access_token: accessToken },
    );
}

export function submitScanningJob(job) {
    const data = {
        name: job.name,
        scannerId: job.scannerId,
        interval: job.interval * secondsPerMinute,
        duration: job.duration / 1000,
    };
    return API.postJSON('/api/scan-jobs', data);
}

export function getScanningJobs() {
    return API.get('/api/scan-jobs').then(jobs => jobs.map(jsonJob => ({
        identifier: jsonJob.identifier,
        name: jsonJob.name,
        scannerId: jsonJob.scannerId,
        interval: new Duration(jsonJob.interval),
        duration: new Duration(jsonJob.duration),
        startTime: jsonJob.startTime && new Date(jsonJob.startTime),
        terminationMessage: jsonJob.terminationMessage,
        terminationTime: jsonJob.terminationTime && new Date(jsonJob.terminationTime),
    })));
}

export function startScanningJob(job) {
    return API.postJSON(`/api/scan-jobs/${job.identifier}/start`, {});
}

export function getScanners() {
    return API.get('/api/scanners')
        .then(r => r.map(scanner => ({
            name: scanner.name,
            identifier: scanner.identifier,
            power: scanner.power,
        })));
}

export function getScannerJob(scannerId) {
    return API.get(`/api/scanners/${scannerId}/job`);
}

export function deleteScanningJob(job) {
    return API.delete(`/api/scan-jobs/${job}`);
}

export function terminateScanningJob(jobid, message) {
    return API.postJSON(`/api/scan-jobs/${jobid}/terminate`, { message });
}

export function extractFeatures(jobId, analysis, keepQC) {
    return API.postJSON(
        '/api/project/feature_extract',
        {
            analysis_directory: `/root/${jobId}/${analysis}`,
            keep_qc: keepQC,
        },
    );
}
