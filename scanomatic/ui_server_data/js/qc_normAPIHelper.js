// var baseUrl = "http://localhost:5000";
const baseUrl = '';
const BrowseRootPath = `${baseUrl}/api/results/browse`;
const NormalizeRefOffsets = `${baseUrl}/api/results/normalize/reference/offsets`;
const NormalizeProjectUrl = `${baseUrl}/api/results/normalize`;

let lock;

function BrowseProjectsRoot(callback) {
    const path = BrowseRootPath;

    d3.json(path, (error, json) => {
        if (error) callback(null);
        else {
            const names = json.names;
            const urls = json.urls;
            const len = names.length;
            const projects = [];
            for (let i = 0; i < len; i++) {
                const projectUrl = urls[i];
                var projectName;
                if (names[i] == null) { projectName = `[${getLastSegmentOfPath(projectUrl)}]`; } else projectName = names[i];
                const project = { name: projectName, url: projectUrl };
                projects.push(project);
            }
            callback(projects);
        }
        return console.warn(error);
    });
}

function BrowsePath(url, callback) {
    const path = baseUrl + url.replace(branchSymbol, '');

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        const names = json.names;
        const urls = json.urls;
        const isProject = json.is_project;
        const len = names.length;
        const paths = [];
        for (let i = 0; i < len; i++) {
            const folder = getLastSegmentOfPath(urls[i]);
            const path = { name: `${names[i]} [${folder}]`, url: urls[i] };
            paths.push(path);
        }
        let projectDetails = '';
        if (isProject) {
            projectDetails = {
                analysis_date: json.analysis_date,
                analysis_instructions: json.analysis_instructions,
                change_date: json.change_date,
                extraction_date: json.extraction_date,
                phenotype_names: json.phenotype_names,
                phenotype_normalized_names: json.phenotype_normalized_names,
                project_name: json.project_name,
                project: json.project,
                add_lock: json.add_lock,
                remove_lock: json.remove_lock,
                export_phenotypes: json.export_phenotypes,
            };
        }
        const browse = { isProject, paths, projectDetails };
        callback(browse);
    });
}

function GetProjectRuns(url, callback) {
    const path = baseUrl + url;

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        const names = json.names;
        const urls = json.urls;
        const len = names.length;
        const projects = [];
        for (let i = 0; i < len; i++) {
            const folder = getLastSegmentOfPath(urls[i]);
            const project = { name: `${names[i]} [${folder}]`, url: urls[i] };
            projects.push(project);
        }
        callback(projects);
    });
}

function GetAPILock(url, callback) {
    if (url) {
        const path = `${baseUrl + url}/${addCacheBuster(true)}`;
        d3.json(path, (error, json) => {
            if (error) return console.warn(error);

            let permissionText;
            let lock;
            if (json.success == true) {
                lock = json.lock_key;
                permissionText = 'Read/Write';
            } else {
                permissionText = 'Read Only';
                lock = null;
            }
            const lockData = {
                lock_key: lock,
                lock_state: permissionText,
            };
            callback(lockData);
        });
    }
}

function addCacheBuster(fisrt) {
    if (fisrt === true) { return `?buster=${Math.random().toString(36).substring(7)}`; }
    return `&buster=${Math.random().toString(36).substring(7)}`;
}

function addKeyParameter(key) {
    if (key) { return `?lock_key=${key}${addCacheBuster()}`; }
    return '';
}

function GetRunPhenotypePath(url, key, callback) {
    const path = baseUrl + url + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        const phenoPath = json.phenotype_names;
        callback(phenoPath);
    });
}

function RemoveLock(url, key, callback) {
    const path = baseUrl + url + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json.success);
    });
}

function GetRunPhenotypes(url, key, callback) {
    const path = baseUrl + url + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        const phenotypes = [];
        for (let i = 0; i < json.phenotypes.length; i++) {
            phenotypes.push({
                name: json.names[i],
                phenotype: json.phenotypes[i],
                url: json.phenotype_urls[i],
            });
        }
        callback(phenotypes);
    });
}

function GetPhenotypesPlates(url, key, callback) {
    const path = baseUrl + url + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        const plates = [];
        for (let i = 0; i < json.urls.length; i++) {
            plates.push({ index: json.plate_indices[i], url: json.urls[i] });
        }
        callback(plates);
    });
}

function GetPlateData(url, isNormalized, metaDataPath, phenotypePlaceholderMetaDataPath, key, callback) {
    const path = baseUrl + url + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        if (json.success === false) {
            alert(`Could not display the data! \n${json.reason}`);
            callback(null);
        } else {
            GetGtPlateData(metaDataPath, phenotypePlaceholderMetaDataPath, key, isNormalized, (gtData) => {
                GetGtWhenPlateData(metaDataPath, phenotypePlaceholderMetaDataPath, key, isNormalized, (gtWhenData) => {
                    GetYieldPlateData(metaDataPath, phenotypePlaceholderMetaDataPath, key, isNormalized, (yieldData) => {
                        const qIdxCols = json.qindex_cols;
                        const qIdxRows = json.qindex_rows;
                        const qIdxSort = [];
                        if (qIdxCols.length === qIdxRows.length) {
                            let idx = 0;
                            for (let i = 0; i < qIdxRows.length; i++) {
                                qIdxSort.push({ idx, row: qIdxRows[i], col: qIdxCols[i] });
                                idx += 1;
                            }
                            window.qc.actions.setQualityIndexQueue(qIdxSort, window.qc.selectors.getPlate());
                        }
                        const plate = {
                            plate_data: json.data,
                            plate_phenotype: json.phenotype,
                            Plate_metadata: {
                                badData: json.BadData,
                                empty: json.Empty,
                                noGrowth: json.NoGrowth,
                                undecidedProblem: json.UndecidedProblem,
                            },
                            Growth_metaData: {
                                gt: isNormalized === true ? null : gtData,
                                gtWhen: isNormalized === true ? null : gtWhenData,
                                yld: isNormalized === true ? null : yieldData,
                            },
                        };
                        callback(plate);
                    });
                });
            });
        }
    });
}

function GetGtPlateData(url, placeholder, key, isNormalized, callback) {
    const path = baseUrl + url.replace(placeholder, 'GenerationTime') + addKeyParameter(key);

    if (isNormalized === true) callback(null);
    console.log(`Metadata GTWhen Path:${path}`);
    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json.data);
    });
}

function GetGtWhenPlateData(url, placeholder, key, isNormalized, callback) {
    const path = baseUrl + url.replace(placeholder, 'GenerationTimeWhen') + addKeyParameter(key);

    if (isNormalized === true) callback(null);
    console.log(`Metadata GTWhen Path:${path}`);
    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json.data);
    });
}

function GetYieldPlateData(url, placeholder, key, isNormalized, callback) {
    const path = baseUrl + url.replace(placeholder, 'ExperimentGrowthYield') + addKeyParameter(key);

    if (isNormalized === true) callback(null);
    console.log(`Metadata yield path:${path}`);
    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json.data);
    });
}

function GetExperimentGrowthData(plateUrl, key, callback) {
    const path = baseUrl + plateUrl + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json);
    });
}

function GetMarkExperiment(plateUrl, key, callback) {
    const path = baseUrl + plateUrl + addKeyParameter(key);

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json);
    });
}

function GetNormalizeProject(projectPath, key, callback) {
    const path = `${NormalizeProjectUrl}/${projectPath}${addKeyParameter(key)}`;

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        callback(json);
    });
}

function GetExport(url, callback) {
    const path = baseUrl + url;

    $.ajax({
        url: path,
        type: 'POST',
        dataType: 'json',
        contentType: 'application/json',
        success(data) {
            console.log(`API:${JSON.stringify(data)}`);
            callback(data);
        },
        error(data) {
            console.log(`API ERROR: ${JSON.stringify(data)}`);
            callback(data);
        },
    });
}

function GetReferenceOffsets(callback) {
    const path = NormalizeRefOffsets;

    d3.json(path, (error, json) => {
        if (error) return console.warn(error);

        const names = json.offset_names;
        const values = json.offset_values;
        const len = names.length;
        const offsets = [];
        for (let i = 0; i < len; i++) {
            const ofset = { name: names[i], value: values[i] };
            offsets.push(ofset);
        }
        callback(offsets);
    });
}

function GetSelectionFromCoordinates(coordinates) {
    const jsonObject = {
        coordinates: [[1, 1], [1, 2], [1, 3]],
    };
    const jsonData = JSON.stringify(jsonObject);

    $.ajax({
        url: 'http://local:5000/api/tools/coordinates/parse',
        type: 'POST',
        data: jsonData,
        dataType: 'json',
        contentType: 'application/json',
        success(data) {
            console.log(`API:${JSON.stringify(data)}`);
        },
        error(data) {
            console.log(`API ERROR: ${JSON.stringify(data)}`);
        },
    });
}
