//var baseUrl = "http://localhost:5000";
var baseUrl = "";
var BrowseRootPath = baseUrl+"/api/results/browse";
var NormalizeRefOffsets = baseUrl+"/api/results/normalize/reference/offsets";
var NormalizeProjectUrl = baseUrl+"/api/results/normalize";

var lock;

function BrowseProjectsRoot(callback) {
    var path = BrowseRootPath;

    d3.json(path, function(error, json) {
        if (error) callback(null);
        else {
            var names = json.names;
            var urls = json.urls;
            var len = names.length;
            var projects = [];
            for (var i = 0; i < len; i++) {
                var projectUrl = urls[i];
                var projectName;
                if (names[i] == null)
                    projectName = "[" + getLastSegmentOfPath(projectUrl) + "]";
                else projectName = names[i];
                var project = { name: projectName, url: projectUrl }
                projects.push(project);
            }
            callback(projects);
        }
        return console.warn(error);
    });
};

function BrowsePath(url, callback) {
    var path = baseUrl + url.replace(branchSymbol, "");

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            var names = json.names;
            var urls = json.urls;
            var isProject = json.is_project;
            var len = names.length;
            var paths = [];
            for (var i = 0; i < len; i++) {
                var folder = getLastSegmentOfPath(urls[i]);
                var path = { name: names[i] + " [" + folder + "]", url: urls[i] }
                paths.push(path);
            }
            var projectDetails="";
            if (isProject) {
                projectDetails = {
                    analysis_date: json.analysis_date,
                    analysis_instructions: json.analysis_instructions,
                    change_date: json.change_date,
                    extraction_date: json.extraction_date,
                    phenotype_names: json.phenotype_names,
                    phenotype_normalized_names : json.phenotype_normalized_names,
                    project_name: json.project_name,
                    project: json.project,
                    add_lock: json.add_lock,
                    remove_lock: json.remove_lock,
                    export_phenotypes: json.export_phenotypes
                }
            }
            var browse = { isProject: isProject, paths: paths, projectDetails: projectDetails }
            callback(browse);
        }
    });
};

function GetProjectRuns(url, callback) {
    var path = baseUrl + url;

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            var names = json.names;
            var urls = json.urls;
            var len = names.length;
            var projects = [];
            for (var i = 0; i < len; i++) {
                var folder = getLastSegmentOfPath(urls[i]);
                var project = { name: names[i] + " ["+folder+"]", url: urls[i] }
                projects.push(project);
            }
            callback(projects);
        }
    });
};

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

function GetRunPhenotypePath(url, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            var phenoPath = json.phenotype_names;
            callback(phenoPath);
        }
    });
};

function RemoveLock(url, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json.success);
        }
    });
};

function GetRunPhenotypes(url, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            var phenotypes = [];
            for (var i = 0; i < json.phenotypes.length; i++) {
                phenotypes.push({
                    name: json.names[i],
                    phenotype: json.phenotypes[i],
                    url: json.phenotype_urls[i],
                });
            }
            callback(phenotypes);
        }
    });
};

function GetPhenotypesPlates(url, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            var plates = [];
            for (var i = 0; i < json.urls.length; i++) {
                plates.push({ index: json.plate_indices[i], url: json.urls[i]});
            }
            callback(plates);
        }
    });
};

function GetPlateData(url, isNormalized, metaDataPath, phenotypePlaceholderMetaDataPath, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            if (json.success === false) {
                alert("Could not display the data! \n" + json.reason);
                callback(null);
            }
            else
                GetGtPlateData(metaDataPath, phenotypePlaceholderMetaDataPath, key, isNormalized, function (gtData) {
                    GetGtWhenPlateData(metaDataPath, phenotypePlaceholderMetaDataPath, key, isNormalized, function (gtWhenData) {
                        GetYieldPlateData(metaDataPath, phenotypePlaceholderMetaDataPath, key, isNormalized, function (yieldData) {
                            var qIdxCols = json.qindex_cols;
                            var qIdxRows = json.qindex_rows;
                            var qIdxSort = [];
                            if (qIdxCols.length === qIdxRows.length) {
                                var idx = 0;
                                for (var i = 0; i < qIdxRows.length ; i++) {
                                    qIdxSort.push({ idx: idx, row: qIdxRows[i], col: qIdxCols[i] });
                                    idx = idx + 1;
                                }
                                window.qc.actions.setQualityIndexQueue(qIdxSort, window.qc.selectors.getPlate());
                            }
                            var plate = {
                                plate_data: json.data,
                                plate_phenotype: json.phenotype,
                                plate_qIdxSort: qIdxSort,
                                Plate_metadata : {
                                    plate_BadData: json.BadData,
                                    plate_Empty: json.Empty,
                                    plate_NoGrowth: json.NoGrowth,
                                    plate_UndecidedProblem: json.UndecidedProblem
                                },
                                Growth_metaData: {
                                    gt: isNormalized === true ? null : gtData,
                                    gtWhen: isNormalized === true ? null : gtWhenData,
                                    yld: isNormalized === true ? null : yieldData
                                }
                            }
                            callback(plate);
                        });
                    });
                });
        }
    });
};

function GetGtPlateData(url, placeholder, key, isNormalized, callback) {
    var path = baseUrl + url.replace(placeholder, "GenerationTime") + addKeyParameter(key);

    if (isNormalized === true) callback(null);
    console.log("Metadata GTWhen Path:" + path);
    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json.data);
        }
    });
};

function GetGtWhenPlateData(url, placeholder, key, isNormalized, callback) {
    var path = baseUrl + url.replace(placeholder, "GenerationTimeWhen") + addKeyParameter(key);

    if (isNormalized === true) callback(null);
    console.log("Metadata GTWhen Path:" + path);
    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json.data);
        }
    });
};

function GetYieldPlateData(url, placeholder, key, isNormalized, callback) {
    var path = baseUrl + url.replace(placeholder, "ExperimentGrowthYield") + addKeyParameter(key);

    if (isNormalized === true) callback(null);
    console.log("Metadata yield path:" + path);
    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json.data);
        }
    });
};

function GetExperimentGrowthData(plateUrl, key, callback) {
    var path = baseUrl + plateUrl + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json);
        }
    });
};

function GetMarkExperiment(plateUrl, key, callback) {
    var path = baseUrl + plateUrl + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json);
        }
    });
};

function GetNormalizeProject(projectPath, key, callback) {
    var path = NormalizeProjectUrl + "/"+ projectPath + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            callback(json);
        }
    });
};

function GetExport(url, callback) {
    var path = baseUrl + url;

    $.ajax({
        url: path,
        type: "POST",
        dataType: "json",
        contentType: 'application/json',
        success: function (data) {
            console.log("API:" + JSON.stringify(data));
            callback(data);
        },
        error: function (data) {
            console.log("API ERROR: " + JSON.stringify(data));
            callback(data);
        }
    });
};

function GetReferenceOffsets(callback) {
    var path = NormalizeRefOffsets;

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);
        else {
            var names = json.offset_names;
            var values = json.offset_values;
            var len = names.length;
            var offsets = [];
            for (var i = 0; i < len; i++) {
                var ofset = { name: names[i], value: values[i] }
                offsets.push(ofset);
            }
            callback(offsets);
        }
    });
};

function GetSelectionFromCoordinates(coordinates) {

    var jsonObject = {
        "coordinates": [[1, 1], [1, 2], [1, 3]]
    };
    var jsonData = JSON.stringify(jsonObject);

    $.ajax({
        url: "http://local:5000/api/tools/coordinates/parse",
        type: "POST",
        data: jsonData,
        dataType: "json",
        contentType: 'application/json',
        success: function (data) {
            console.log("API:" + JSON.stringify(data));
        },
        error: function (data) {
            console.log("API ERROR: " + JSON.stringify(data));
        }
    });
}
