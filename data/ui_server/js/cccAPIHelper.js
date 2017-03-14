var baseUrl = "http://localhost:5000";
//var baseUrl = "";
var InitiateCCCPath = baseUrl + "/api/calibration/initiate_new";
var GetFixtruesPath = baseUrl + "/api/data/fixture/names";
var GetPinningFormatsPath = baseUrl + "/api/analysis/pinning/formats";
var GetMarkersPath = baseUrl + "/api/data/markers/detect/";
var GetTranposedMarkerPath = baseUrl + "/api/data/fixture/calculate/";
var GetGrayScaleAnalysisPath = baseUrl + "/api/data/grayscale/image/";
var GetImageId_prePath = baseUrl + "/api/calibration/";
var GetImageId_postPath = "/add_image";

var lock;

function GetFixtures(callback) {
    var path = GetFixtruesPath;

    d3.json(path, function(error, json) {
        if (error) console.warn(error);
        else {
            var fixtrues = json.fixtures;
            callback(fixtrues);
        }
    });
};

function GetPinningFormats(callback) {
    var path = GetPinningFormatsPath;

    d3.json(path, function (error, json) {
        if (error) console.warn(error);
        else {
            var fixtrues = json.pinning_formats;
            callback(fixtrues);
        }
    });
};

function InitiateCCC(species, reference, successCallback, errorCallback) {
    var path = InitiateCCCPath;
    var formData = new FormData();
    formData.append("species", species);
    formData.append("reference", reference);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback
    });
}

function GetGrayScaleAnalysis(grayScaleName, imageData, successCallback, errorCallback) {
    var path = GetGrayScaleAnalysisPath + grayScaleName;
    var formData = new FormData();
    formData.append("image", imageData);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback
    });
}

function GetImageId(cccId, file, accessToken, successCallback, errorCallback) {
    var path = GetImageId_prePath + cccId + GetImageId_postPath;
    var formData = new FormData();
    formData.append("image_data", file);
    formData.append("access_token", accessToken);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback
    });
}

function GetMarkers(fixtureName, file, successCallback, errorCallback) {
    var path = GetMarkersPath + fixtureName;
    var formData = new FormData();
    formData.append("image", file);
    formData.append("save", "false");
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback
    });
}

function GetTransposedMarkersV2(fixtureName, markers, file, successCallback, errorCallback) {
    var path = GetTranposedMarkerPath + fixtureName;
    var formData = new FormData();
    formData.append("image", file);
    formData.append("markers", markers);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback
    });
}

function GetTransposedMarkers(fixtureName, markers, successCallback, errorCallback) {
    var path = GetTranposedMarkerPath + fixtureName;
    var formData = new FormData();
    formData.append("markers", markers);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: successCallback,
        error: errorCallback
    });
}

function GetAPILock(url, callback) {
    if (url) {
        var path = baseUrl + url+"/" + addCacheBuster(true);
        d3.json(path, function (error, json) {
            if (error) return console.warn(error);
            else {
                var permissionText;
                var lock;
                if (json.success == true) {
                    lock = json.lock_key;
                    permissionText = "Read/Write";
                }
                else {
                    permissionText = "Read Only";
                    lock = null;
                }
                var lockData = {
                    lock_key: lock,
                    lock_state: permissionText
                }
                callback(lockData);
            }
        });
    }
};

function addCacheBuster(fisrt) {
    if(fisrt===true)
        return "?buster=" + Math.random().toString(36).substring(7);
    return "&buster=" + Math.random().toString(36).substring(7);
}

function addKeyParameter(key) {
    if (key)
        return "?lock_key=" + key + addCacheBuster();
    else
        return "";
}

function RemoveLock(url, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json.success);
        }
    });
};

