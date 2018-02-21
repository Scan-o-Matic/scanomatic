let spinner = null;
let spinTarget = null;
const selRunNormPhenotypesName = 'selRunNormPhenotypes';
const selRunPhenotypesName = 'selRunPhenotypes';
const dispatch = d3.dispatch('setExp', 'reDrawExp');
const branchSymbol = '¤';
let qIndexQueue = [];
let qIndexCurrent = 0;
const qIdxOperations = {
    Current: 0,
    Prev: -1,
    Next: 1,
};

function initSpinner() {
    spinTarget = document.getElementById('divLoading');
    spinner = new Spinner({
        lines: 9, // The number of lines to draw
        length: 9, // The length of each line
        width: 5, // The line thickness
        radius: 20, // The radius of the inner circle
        color: '#000000', // #rgb or #rrggbb or array of colors
        speed: 1.9, // Rounds per second
        trail: 40, // Afterglow percentage
        className: 'spinner', // The CSS class to assign to the spinner
    }).spin(spinTarget);
}
﻿
function getUrlParameter(sParam) {
    var sPageUrl = decodeURIComponent(window.location.search.substring(1));
    var sUrlVariables = sPageUrl.split('&');
    var sParameterName, i;

    for (i = 0; i < sUrlVariables.length; i++) {
        sParameterName = sUrlVariables[i].split("=");

        if (sParameterName[0] === sParam) {
            return sParameterName[1] === undefined ? true : sParameterName[1];
        }
    }
    return null;
};

function FillPlate() {
    //32x48
    var count = 0;
    var plate = d3.range(48).map(function () {
        return d3.range(32).map(function () {
            count += 1;
            return count;
        });
    });
    return plate;
}

function getLastSegmentOfPath(path) {
    var parts = path.split("/");
    var lastPart = parts.pop();
    if (lastPart === "")
        lastPart = parts.pop();
    return lastPart;
}

function getExtentFromMultipleArrs() {
    if (!arguments.length) return null;
    var extremeValues = [];
    for (var i = 0; i < arguments.length; i++) {
        extremeValues.push(d3.max(arguments[i]));
        extremeValues.push(d3.min(arguments[i]));
    }
    var ext = d3.extent(extremeValues);
    return ext;
}

function getBaseLog(base, value) {
    return Math.log(value) / Math.log(base);
}

function wait() {
    $("#divLoading")
        .html('<p>Talking to server...</p>')
        .modal({ escapeClose: false, clickClose: false, showClose: false });
    spinner.spin(spinTarget);
}

function modalMessage(msg, allowClose) {
    $('#divLoading')
        .html(`<p>${msg}</p>`)
        .modal({ escapeClose: !!allowClose, clickClose: !!allowClose, showClose: false });
}

function getLock(callback) {
    var lockPath = $("#spLock").data("lock_path");
    wait();
    GetAPILock(lockPath,
        function (lockData) {
            if (lockData != null) {
                var lock = lockData.lock_key;
                var permisssionText = lockData.lock_state;
                $("#spLock").text(permisssionText);
                $("#spLock").data("lock_key", lock);
                $("#tbProjectDetails").show();
                callback();
                stopWait();
            }
        });
}

function fillProjectDetails(projectDetails) {
    console.log("Project details:" + projectDetails.project);
    $("#spProject_name").text(projectDetails.project_name);
    $("#spProject").text(projectDetails.project);
    $("#spExtraction_date").text(projectDetails.extraction_date);
    $("#spAnalysis_date").text(projectDetails.analysis_date);
    $("#spAnalysis_instructions").text(projectDetails.analysis_instructions);
    $("#spLock").data("lock_path", projectDetails.add_lock);
    $("#spLock").data("unLock_path", projectDetails.remove_lock);
    $("#spQidx").data("qIdx", "");
    var inner = "";
    inner += "<li><a href='" + baseUrl + "/api/results/export/phenotypes/Absolute/" + projectDetails.project + "'>Absolute</a></li>";
    inner += "<li><a href='" + baseUrl + "/api/results/export/phenotypes/NormalizedRelative/" + projectDetails.project + "'>Normalized Relative</a></li>";
    inner += "<li><a href='" + baseUrl + "/api/results/export/phenotypes/NormalizedAbsoluteBatched/" + projectDetails.project + "'>Normalized Absolute Batched</a></li>";
    $("#ulExport").html(inner);
    getLock(function () {
        $("#btnUploadMetaData").click(function () {
            var key = $("#spLock").data("lock_key");
            var addMetaDataUrl = baseUrl + "/api/results/meta_data/add/" + projectDetails.project + addKeyParameter(key);
            var file = $("#meta_data")[0].files[0];
            if (!file) {
                alert("You need to select a valid file!");
                return;
            }
            var extension = file.name.split('.').pop();
            var formData = new FormData();
            formData.append("meta_data", file);
            formData.append("file_suffix", extension);
            $.ajax({
                url: addMetaDataUrl,
                type: "POST",
                contentType: false,
                enctype: 'multipart/form-data',
                data: formData,
                processData: false,
                success: function (data) {
                    if (data.success == true)
                        alert("The data was uploaded successfully:");
                    else
                        alert("There was a problem with the upload: " + data.reason);
                },
                error: function (data) {
                    alert("error:" + data.responseText);
                }
            });
        });
        $("#btnBrowseProject").click();
        drawRunPhenotypeSelection(projectDetails.phenotype_names);
        drawRunNormalizedPhenotypeSelection(projectDetails.phenotype_normalized_names);
        drawReferenceOffsetSelecton();
    });
}

function setExperimentByQidx(operation) {
    var queue = getQIndexCoord(operation);
    var row = queue.row;
    var col = queue.col;
    dispatch.setExp("id" + row + "_" + col);
}

function setExperimentByCoord(row, col) {
    dispatch.setExp("id" + row + "_" + col);
}

function isQualityControlOn() {
    var on = $("#divMarkStates").is(':visible');
    return on;
}

function getQIndexCoord(operation) {
    qIndexCurrent = qIndexCurrent + operation;
    var maxQueueSize = qIndexQueue.length - 1;
    if (qIndexCurrent < 0)
        qIndexCurrent = maxQueueSize;
    if (qIndexCurrent > maxQueueSize)
        qIndexCurrent = 0;
    var item = qIndexQueue[qIndexCurrent];
    return item;
}

function getChar(event) {
    if (event.which == null) {
        return String.fromCharCode(event.keyCode); // IE
    } else if (event.which != 0 && event.charCode != 0) {
        return String.fromCharCode(event.which); // the rest
    } else {
        return ''; // special key
    }
}

function stopWait() {
    spinner.stop();
    $.modal.close();
}

function getLock_key() {
    return $("#spLock").data("lock_key");
}

function isPlateAllNull(plateData) {
    for (var i = 0, len = plateData.length; i < len; i += 1)
        for (var j = 0, lenj = plateData[1].length; j < lenj; j += 1)
            if (plateData[i][j] !== null)
                return false;

    return true;
}

function createMarkButton(buttonId, type) {
    var btn = d3.select(buttonId)
        .append("svg")
        .attr({
            "width": 25,
            "height": 25
        });
    addSymbolToSGV(btn, type);
    btn.append("use")
        .attr({
            "xlink:href": "#" + getValidSymbol(type),
            "x": 0,
            "y": 0,
            "width": 25,
            "height": 25
        });
};

function createMarkButtons() {
    createMarkButton("#btnMarkOK", plateMetaDataType.OK);
    createMarkButton("#btnMarkBad", plateMetaDataType.BadData);
    createMarkButton("#btnMarkEmpty", plateMetaDataType.Empty);
    createMarkButton("#btnMarkNoGrowth", plateMetaDataType.NoGrowth);
}

function projectSelectionStage(level) {
    switch (level) {
        case "project":
            $("#displayArea").hide();
            $("#dialogGrid").hide();
            $(".loPhenotypeSelection").hide();
            $(".loPlateSelection").hide();
            $("#tbProjectDetails").hide();
            $("#divMarkStates").hide();
            $("#qidxHint").hide();

            break;
        case "Phenotypes":
            $(".loPhenotypeSelection").show();
            break;
        case "Plates":
            $("#displayArea").show();
            $(".loPlateSelection").show();
            d3.select("#selPhenotypePlates").remove();
            break;
        default:
    }
}

//Mark Selected Experiment
function markExperiemnt(mark, all) {
    if (!isQualityControlOn()) return;
    var plateIdx = $("#currentSelection").data("plateIdx");
    var row = $("#currentSelection").data("row");
    var col = $("#currentSelection").data("col");
    var phenotype = $("#currentSelection").data("phenotype");
    var project = $("#currentSelection").data("project");
    // /api/results/curve_mark/set/<mark>/<phenotype>/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>"
    var path = "";
    if( all !== true)
        path = "/api/results/curve_mark/set/" + mark + "/" + phenotype + "/" + plateIdx + "/" + row + "/" + col + "/" + project;
    else
        path = "/api/results/curve_mark/set/" + mark + "/" + plateIdx + "/" + row + "/" + col + "/" + project;
    var lockKey = getLock_key();
    wait();
    GetMarkExperiment(path, lockKey, function (gData) {
        if (gData.success === true) {
            dispatch.reDrawExp("id" + row + "_" + col, mark);
            var queueCurrent = getQIndexCoord(qIdxOperations.Current);
            if (queueCurrent.row == row && queueCurrent.col == col)
                setExperimentByQidx(qIdxOperations.Next);
            else
                setExperimentByQidx(qIdxOperations.Current);
        }
        else
            alert(gData.success + " : " + gData.reason);
        stopWait();
    });
}


function nodeCollapse() {
    //alert("Collapsed: " + this.id);
}

function nodeExpand() {
    var parentId = this.id;
    BrowsePath(parentId, function (browse) {
        console.log("ParentID:" + parentId);
        console.log("is project:" + browse.isProject);
        console.log("paths len:" + browse.paths.length);
        var parentNode = $("#tblProjects").treetable("node", parentId);
        var nodeToAdd;
        var row;
        if (!browse.isProject && browse.paths.length === 0) {
            nodeToAdd = $("#tblProjects").treetable("node", "empty" + parentId);
            if (!nodeToAdd) {
                row = '<tr data-tt-id="empty' + parentId + '" data-tt-parent-id="' + parentId + '" >';
                row += "<td><span class='file'>This project is Empty ...</span></td>";
                row += "</tr>";
                $("#tblProjects").treetable("loadBranch", parentNode, row);
            }
        } else if (!browse.isProject) {
            var rows = "";
            $.each(browse.paths,
                function (key, value) {
                    row = '<tr data-tt-id="' + branchSymbol + value.url + '" data-tt-parent-id="' + parentId + '" data-tt-branch="true" >';
                    row += "<td>" + value.name + "</td>";
                    row += "</tr>";
                    rows += row;
                });
            $("#tblProjects").treetable("loadBranch", parentNode, rows);
            console.log("addedNodes rows:" + rows);
        } else if (browse.isProject) {
            nodeToAdd = $("#tblProjects").treetable("node", "project" + parentId);
            if (!nodeToAdd) {
                row = '<tr data-tt-id="project' + parentId + '" data-tt-parent-id="' + parentId + '"  >';
                console.log("button id: " + browse.projectDetails);
                row += "<td><button id='" + browse.projectDetails.project + "'>Here is your project</button></td>";
                row += "</tr>";
                $("#tblProjects").treetable("loadBranch", parentNode, row);
                var btn = $(document.getElementById(browse.projectDetails.project));
                btn.on("click", function () { fillProjectDetails(browse.projectDetails); });
                btn.attr("class", "attached");
            }
        }
    });
    console.log("Expanded: " + this.id);
}

//draw Reference Offset selection
function drawReferenceOffsetSelecton() {
    var elementName = "selRefOffSets";
    GetReferenceOffsets(function (offsets) {
        d3.select("#" + elementName).remove();
        var selPhen = d3.select("#divReferenceOffsetSelector")
            .append("select")
            .attr("id", elementName);
        var options = selPhen.selectAll("optionPlaceholders")
            .data(offsets)
            .enter()
            .append("option");
        options.attr("value", function (d) { return d.value; });
        options.text(function (d) { return d.name; });
        $("#" + elementName).selectedIndex = 0;
        drawPhenotypePlatesSelection();
    });
};

//draw run phenotypes selection
function drawRunPhenotypeSelection(path) {
    projectSelectionStage("Phenotypes");
    console.log("Phenotypes path: " + path);
    var lockKey = getLock_key();
    GetRunPhenotypes(path, lockKey, function (runPhenotypes) {
        d3.select("#" + selRunPhenotypesName).remove();
        var selPhen = d3.select("#divRunPhenotypesSelector")
            .append("select")
            .attr("id", selRunPhenotypesName);
        var options = selPhen.selectAll("optionPlaceholders")
            .data(runPhenotypes)
            .enter()
            .append("option");
        options.attr("value", function(d) {return d.url;});
        options.text(function (d) { return d.name; });
        selPhen.on("change", drawPhenotypePlatesSelection);
        $("#" + selRunPhenotypesName).selectedIndex = 0;
        //drawPhenotypePlatesSelection();
    });
};

function drawRunNormalizedPhenotypeSelection(path) {
    console.log("Norm Phenotypes path: " + path);
    var lockKey = getLock_key();
    GetRunPhenotypes(path, lockKey, function (runPhenotypes) {
        d3.select("#" + selRunNormPhenotypesName).remove();
        var selPhen = d3.select("#divRunPhenotypesSelector")
            .append("select")
            .attr("id", selRunNormPhenotypesName);
        var options = selPhen.selectAll("optionPlaceholders")
            .data(runPhenotypes)
            .enter()
            .append("option");
        options.attr("value", function(d) {return d.url;});
        options.text(function (d) { return d.name; });
        selPhen.on("change", drawPhenotypePlatesSelection);
        $("#" + selRunNormPhenotypesName).toggle();
    });
};

//draw phenotypes plates selection
function drawPhenotypePlatesSelection() {
    var isNormalized = $("#ckNormalized").is(':checked');
    var selectedPhen = $("#" + selRunPhenotypesName).val();
    var selectedNromPhen = $("#" + selRunNormPhenotypesName).val();
    var path = isNormalized ? selectedNromPhen : selectedPhen;
    if (!path)
        return;
    projectSelectionStage("Plates");
    console.log("plates: " + path);
    var lockKey = getLock_key();
    GetPhenotypesPlates(path, lockKey, function (phenotypePlates) {
        //plate buttons
        d3.selectAll(".plateSelectionButton").remove();
        var selPlates = d3.select("#divPhenotypePlatesSelecton");
        var buttons = selPlates.selectAll("buttonPlaceholders")
            .data(phenotypePlates)
            .enter()
            .append("a");
        buttons.attr({
            "type": "button",
            "class": "btn btn-default plateSelectionButton",
            "id": function (d) { return "btnPlate" + d.index },
            "href": "#",
            "role": "button"
        });
        buttons.on("click", function (d) { renderPlate(d) });;
        buttons.text(function (d) { return "Plate " + (d.index + 1); });
        //griding buton
        $("#divPhenotypePlatesSelecton")
            .append("<a type='button' class='btn btn-default btn-xs plateSelectionButton' id='btnShowGrid' href='#' role='button'>Show Grid</a>");
        $("#btnShowGrid").click(showGrid);
        //check for plate index or load plate 0 by default
        var plateIdx = $("#currentSelection").data("plateIdx");
        var plateId = "btnPlate0";
        if (plateIdx) plateId = "btnPlate" + plateIdx;
        //$("#"+plateId).focus();
        document.getElementById(plateId).click();
    });
};

function showGrid() {
    var plateIdx = $("#currentSelection").data("plateIdx");
    var project = $("#currentSelection").data("project");
    var path = "/api/results/gridding/" + plateIdx + "/" + project;
    $("#imgGridding").attr("src", baseUrl + path);
    $("#dialogGrid").show();
    $("#dialogGrid").dialog();
};

//draw plate
function renderPlate(phenotypePlates) {
    var path = phenotypePlates.url;
    var plateIdx = phenotypePlates.index;
    var project = $("#spProject").text();
    console.log("experiment: " + path);
    $("#currentSelection").data("plateIdx", plateIdx);
    $("#currentSelection").data("project", project);
    $("#spnPlateIdx").text((plateIdx+1));
    wait();
    // e.g. /api/results/phenotype/GenerationTimeWhen/1/by4742_h/analysis
    var isNormalized = $("#ckNormalized").is(':checked');
    var phenotypePath = isNormalized ? "/api/results/normalized_phenotype/###/" : "/api/results/phenotype/###/";
    var metaDataPath = phenotypePath + plateIdx + "/" + project;
    var lockKey = getLock_key();
    GetPlateData(path, isNormalized, metaDataPath, "###", lockKey, function (data) {
        $("#plate").empty();
        var allNull = isPlateAllNull(data.plate_data);
        if (!data || allNull) {
            stopWait();
            return;
        }
        var plateData = data.plate_data;
        var plateMetaData = data.Plate_metadata;
        var growthMetaData = data.Growth_metaData;
        var phenotypeName = data.plate_phenotype;
        qIndexQueue = data.plate_qIdxSort;
        var plate = DrawPlate("#plate", plateData, growthMetaData, plateMetaData, phenotypeName, dispatch);
        var row = $("#currentSelection").data("row");
        var col = $("#currentSelection").data("col");
        if (row && col)
            setExperimentByCoord(row, col);
        stopWait();
        plate.on("SelectedExperiment", function (datah) {
                console.log("dispatched:" + datah.coord);
                var arr = datah.coord.split(",");
                var row = arr[0];
                var col = arr[1];
                $("#currentSelection").data("expId", datah.id);
                $("#currentSelection").data("plateIdx", plateIdx);
                $("#currentSelection").data("row", row);
                $("#currentSelection").data("col", col);
                $("#currentSelection").data("phenotype", datah.phenotype);
                $("#currentSelection").data("project", $("#spProject").text());
                // e.g. /api/results/curves/1/31/0/Martin_wt1/analysis
                var expPath = "/api/results/curves/" + plateIdx + "/" + row + "/" + col + "/" + project;
                console.log("curve path:" + expPath);
                var lockKey = getLock_key();
                GetExperimentGrowthData(expPath, lockKey, function (gData) {
                    $("#graph").empty();
                    DrawCurves("#graph", gData, datah.metaDataGt, datah.metaDataGtWhen, datah.metaDataYield);
                });
            });
    });
};
