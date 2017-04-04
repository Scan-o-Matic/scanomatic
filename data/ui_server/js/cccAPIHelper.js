var baseUrl = "http://localhost:5000";
//var baseUrl = "";
var GetSliceImagePath = baseUrl + "/api/calibration/#0#/image/#1#/slice/get/#2#";
var InitiateCCCPath = baseUrl + "/api/calibration/initiate_new";
var GetFixtruesPath = baseUrl + "/api/data/fixture/names";
var GetFixtruesDataPath = baseUrl + "/api/data/fixture/get/";
var GetPinningFormatsPath = baseUrl + "/api/analysis/pinning/formats";
var GetMarkersPath = baseUrl + "/api/data/markers/detect/";
var GetTranposedMarkerPath = baseUrl + "/api/data/fixture/calculate/";
var GetGrayScaleAnalysisPath = baseUrl + "/api/data/grayscale/image/";
var GetImageId_Path = baseUrl + "/api/calibration/#0#/add_image";
var SetCccImageDataPath = baseUrl + "/api/calibration/#0#/image/#1#/data/set";
var SetCccImageSlicePath = baseUrl + "/api/calibration/#0#/image/#1#/slice/set";
var SetGrayScaleImageAnalysisPath = baseUrl + "/api/calibration/#0#/image/#1#/grayscale/analyse";
var SetGrayScaleTransformPath = baseUrl + "/api/calibration/#0#/image/#1#/plate/#2#/transform";
var SetGriddingPath = baseUrl + "/api/calibration/#0#/image/#1#/plate/#2#/grid/set";
var SetColonyDetectionPath = baseUrl + "/api/data/calibration/#0#/image/#1#/plate/#2#/detect/colony/#3#/#4#";
var SetColonyCompressionPath = baseUrl + "/api/data/calibration/#0#/image/#1#/plate/#2#/compress/colony/#3#/#4#";


function GetSliceImageURL(cccId, imageId, slice) {
    var path = GetSliceImagePath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", slice);
    return path;
}

function GetSliceImage(cccId, imageId, slice, successCallback, errorCallback) {
    var path = GetSliceImagePath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", slice);

    $.get(path, successCallback).fail(errorCallback);
}


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

function GetFixtureData(fixtureName, callback) {
    var path = GetFixtruesDataPath + fixtureName;

    d3.json(path, function (error, json) {
        if (error) console.warn(error);
        else {
            callback(json);
        }
    });
};

function GetFixturePlates(fixtureName, callback) {

    GetFixtureData(fixtureName, function (data) {
        var plates = data.plates;
        callback(plates);
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

function SetCccImageData(cccId, imageId, accessToken, dataArray, fixture, successCallback, errorCallback) {
    var path = SetCccImageDataPath.replace("#0#", cccId).replace("#1#", imageId);
    var formData = new FormData();
    formData.append("ccc_identifier", cccId);
    formData.append("image_identifier", imageId);
    formData.append("access_token", accessToken);
    formData.append("fixture", fixture);
    for (var i = 0; i < dataArray.length; i++) {
        var item = dataArray[i];
        formData.append(item.key, item.value);
    }
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

function SetCccImageSlice(cccId, imageId, accessToken, successCallback, errorCallback) {
    var path = SetCccImageSlicePath.replace("#0#", cccId).replace("#1#", imageId);
    var formData = new FormData();
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

function SetGrayScaleImageAnalysis(cccId, imageId, accessToken, successCallback, errorCallback) {
    var path = SetGrayScaleImageAnalysisPath.replace("#0#", cccId).replace("#1#", imageId);
    var formData = new FormData();
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

function SetGrayScaleTransform(scope, cccId, imageId, plate, accessToken, successCallback, errorCallback) {
    var path = SetGrayScaleTransformPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate);
    var formData = new FormData();
    formData.append("access_token", accessToken);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function(data) {
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function SetGridding(scope, cccId, imageId, plate, pinningFormat, accessToken, successCallback, errorCallback) {
    var path = SetGriddingPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate);

    var formData = new FormData();
    formData.append("pinning_format", pinningFormat);
    formData.append("access_token", accessToken);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function (data) {
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function SetColonyDetection(scope, cccId, imageId, plate, accessToken, row, col, successCallback, errorCallback) {
    var path = SetColonyDetectionPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate).replace("#3#", row).replace("#4#", col);

    var formData = new FormData();
    formData.append("access_token", accessToken);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function (data) {
            successCallback(data, scope, row, col);
        },
        error: errorCallback
    });
}

function SetColonyCompression(scope, cccId, imageId, plate, accessToken, colony, row, col, successCallback, errorCallback) {
    var path = SetColonyCompressionPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate).replace("#3#", row).replace("#4#", col);

    var formData = new FormData();
    formData.append("access_token", accessToken);
    formData.append("image", colony.image);
    formData.append("blob", colony.blob);
    formData.append("background", colony.background);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function (data) {
            scope.ColonyRow = row;
            scope.ColonyCol = col;
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function SetColonyCompressionV2(scope, cccId, imageId, plate, accessToken, colony, row, col, successCallback, errorCallback) {
    var path = SetColonyCompressionPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate).replace("#3#", row).replace("#4#", col);

    var data = {
        access_token: accessToken,
        image: colony.image,
        blob: colony.blob,
        background: colony.background
    };
    $.ajax({
        url: path,
        method: "POST",
        data: JSON.stringify(data),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            scope.ColonyRow = row;
            scope.ColonyCol = col;
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function GetImageId(cccId, file, accessToken, successCallback, errorCallback) {
    var path = GetImageId_Path.replace("#0#", cccId);
    var formData = new FormData();
    formData.append("image", file);
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

