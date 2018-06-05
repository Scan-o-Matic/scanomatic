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
    Reset: 'reset',
    Goto: 'goto',
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
    $('#divLoading')
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
    const lockPath = $('#spLock').data('lock_path');
    wait();
    GetAPILock(lockPath,
        (lockData) => {
            if (lockData != null) {
                const lock = lockData.lock_key;
                const permisssionText = lockData.lock_state;
                $('#spLock').text(permisssionText);
                $('#spLock').data('lock_key', lock);
                $('#tbProjectDetails').show();
                callback();
                stopWait();
            }
        });
}

function fillProjectDetails(projectDetails) {
    console.log("Project details:" + projectDetails.project);
    window.qc.actions.setProject(projectDetails.project);
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

function getQIndexFromCoord(row, col) {
    return qIndexQueue.filter(e => e.row === row && e.col === col)[0].idx;
}

function setExperimentByCoord(row, col) {
    dispatch.setExp(`id${row}_${col}`);
}

function isQualityControlOn() {
    return $('#ckMarkExperiments').is(':checked');
}

function updateQIndexLabel(qIndex) {
    $('#qIndexCurrent').text(qIndex + 1);
}

function updateQIndexCoord(operation, index) {
    if (operation === qIdxOperations.Reset) {
        qIndexCurrent = 0;
    } else if (operation === qIdxOperations.Goto) {
        qIndexCurrent = index;
    } else {
        qIndexCurrent += operation;

        const qIndexMax = qIndexQueue.length - 1;
        if (qIndexCurrent < 0) {
            qIndexCurrent = qIndexMax;
        } else if (qIndexCurrent > qIndexMax) {
            qIndexCurrent = 0;
        }
    }

    updateQIndexLabel(qIndexCurrent);
    return qIndexQueue[qIndexCurrent];
}

function setExperimentByQidx(operation) {
    const queueCurrent = updateQIndexCoord(operation);
    const row = queueCurrent.row;
    const col = queueCurrent.col;
    dispatch.setExp(`id${row}_${col}`);
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
    return $('#spLock').data('lock_key');
}

function isPlateAllNull(plateData) {
    for (let i = 0, len = plateData.length; i < len; i += 1) {
        for (let j = 0, lenj = plateData[1].length; j < lenj; j += 1) {
            if (plateData[i][j] !== null) { return false; }
        }
    }

    return true;
}

function createMarkButton(buttonId, type, oneOnly) {
    const btn = d3.select(buttonId)
        .append('svg')
        .attr({
            width: 25,
            height: 25,
        });
    addSymbolToSGV(btn, type);
    btn.append('use')
        .attr({
            'xlink:href': `#${getValidSymbol(type)}`,
            x: 0,
            y: 0,
            width: 25,
            height: 25,
        });
    if (oneOnly) {
        $(buttonId).append('<div class="mark-this-phenotype">1</div>');
    }
}

function createMarkButtons() {
    createMarkButton('#btnMarkOK', plateMetaDataType.OK);
    createMarkButton('#btnMarkOKOne', plateMetaDataType.OK, true);
    createMarkButton('#btnMarkBad', plateMetaDataType.BadData);
    createMarkButton('#btnMarkEmpty', plateMetaDataType.Empty);
    createMarkButton('#btnMarkNoGrowth', plateMetaDataType.NoGrowth);
}

function projectSelectionStage(level) {
    switch (level) {
    case 'project':
        $('#displayArea').hide();
        $('#dialogGrid').hide();
        $('.loPhenotypeSelection').hide();
        $('.loPlateSelection').hide();
        $('#tbProjectDetails').hide();

        break;
    case 'Phenotypes':
        $('.loPhenotypeSelection').show();
        break;
    case 'Plates':
        $('#displayArea').show();
        $('.loPlateSelection').show();
        d3.select('#selPhenotypePlates').remove();
        break;
    default:
    }
}

// Mark Selected Experiment
function markExperiment(mark, all) {
    if (!isQualityControlOn()) return;
    const plateIdx = $('#currentSelection').data('plateIdx');
    const row = $('#currentSelection').data('row');
    const col = $('#currentSelection').data('col');
    const phenotype = $('#currentSelection').data('phenotype');
    const project = $('#currentSelection').data('project');
    // /api/results/curve_mark/set/<mark>/<phenotype>/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>"
    let path = '';
    if (all !== true) { path = `/api/results/curve_mark/set/${mark}/${phenotype}/${plateIdx}/${row}/${col}/${project}`; } else { path = `/api/results/curve_mark/set/${mark}/${plateIdx}/${row}/${col}/${project}`; }
    const lockKey = getLock_key();
    wait();
    GetMarkExperiment(path, lockKey, (gData) => {
        if (gData.success === true) {
            dispatch.reDrawExp(`id${row}_${col}`, mark);
            const queueCurrent = updateQIndexCoord(qIdxOperations.Current);
            if (queueCurrent.row == row && queueCurrent.col == col) {
                setExperimentByQidx(qIdxOperations.Next);
            } else {
                setExperimentByQidx(qIdxOperations.Current);
            }
        } else { alert(`${gData.success} : ${gData.reason}`); }
        stopWait();
    });
}


function nodeCollapse() {
    // alert("Collapsed: " + this.id);
}

function nodeExpand() {
    const parentId = this.id;
    BrowsePath(parentId, (browse) => {
        console.log(`ParentID:${parentId}`);
        console.log(`is project:${browse.isProject}`);
        console.log(`paths len:${browse.paths.length}`);
        const parentNode = $('#tblProjects').treetable('node', parentId);
        let nodeToAdd;
        let row;
        if (!browse.isProject && browse.paths.length === 0) {
            nodeToAdd = $('#tblProjects').treetable('node', `empty${parentId}`);
            if (!nodeToAdd) {
                row = `<tr data-tt-id="empty${parentId}" data-tt-parent-id="${parentId}" >`;
                row += "<td><span class='file'>This project is Empty ...</span></td>";
                row += '</tr>';
                $('#tblProjects').treetable('loadBranch', parentNode, row);
            }
        } else if (!browse.isProject) {
            let rows = '';
            $.each(browse.paths,
                (key, value) => {
                    row = `<tr data-tt-id="${branchSymbol}${value.url}" data-tt-parent-id="${parentId}" data-tt-branch="true" >`;
                    row += `<td>${value.name}</td>`;
                    row += '</tr>';
                    rows += row;
                });
            $('#tblProjects').treetable('loadBranch', parentNode, rows);
            console.log(`addedNodes rows:${rows}`);
        } else if (browse.isProject) {
            nodeToAdd = $('#tblProjects').treetable('node', `project${parentId}`);
            if (!nodeToAdd) {
                row = `<tr data-tt-id="project${parentId}" data-tt-parent-id="${parentId}"  >`;
                console.log(`button id: ${browse.projectDetails}`);
                row += `<td><button id='${browse.projectDetails.project}'>Here is your project</button></td>`;
                row += '</tr>';
                $('#tblProjects').treetable('loadBranch', parentNode, row);
                const btn = $(document.getElementById(browse.projectDetails.project));
                btn.on('click', () => { fillProjectDetails(browse.projectDetails); });
                btn.attr('class', 'attached');
            }
        }
    });
    console.log(`Expanded: ${this.id}`);
}

// draw Reference Offset selection
function drawReferenceOffsetSelecton() {
    const elementName = 'selRefOffSets';
    GetReferenceOffsets((offsets) => {
        d3.select(`#${elementName}`).remove();
        const selPhen = d3.select('#divReferenceOffsetSelector')
            .append('select')
            .attr('id', elementName);
        const options = selPhen.selectAll('optionPlaceholders')
            .data(offsets)
            .enter()
            .append('option');
        options.attr('value', d => d.value);
        options.text(d => d.name);
        $(`#${elementName}`).selectedIndex = 0;
        drawPhenotypePlatesSelection();
    });
}

// draw run phenotypes selection
function drawRunPhenotypeSelection(path) {
    projectSelectionStage('Phenotypes');
    console.log(`Phenotypes path: ${path}`);
    const lockKey = getLock_key();
    GetRunPhenotypes(path, lockKey, (runPhenotypes) => {
        d3.select(`#${selRunPhenotypesName}`).remove();
        const selPhen = d3.select('#divRunPhenotypesSelector')
            .append('select')
            .attr('id', selRunPhenotypesName);
        const options = selPhen.selectAll('optionPlaceholders')
            .data(runPhenotypes)
            .enter()
            .append('option');
        options.attr('value', d => d.url);
        options.text(d => d.name);
        selPhen.on('change', drawPhenotypePlatesSelection);
        $(`#${selRunPhenotypesName}`).selectedIndex = 0;
        drawPhenotypePlatesSelection();
    });
}

function drawRunNormalizedPhenotypeSelection(path) {
    console.log(`Norm Phenotypes path: ${path}`);
    const lockKey = getLock_key();
    GetRunPhenotypes(path, lockKey, (runPhenotypes) => {
        d3.select(`#${selRunNormPhenotypesName}`).remove();
        const selPhen = d3.select('#divRunPhenotypesSelector')
            .append('select')
            .attr('id', selRunNormPhenotypesName);
        const options = selPhen.selectAll('optionPlaceholders')
            .data(runPhenotypes)
            .enter()
            .append('option');
        options.attr('value', d => d.url);
        options.text(d => d.name);
        selPhen.on('change', drawPhenotypePlatesSelection);
        $(`#${selRunNormPhenotypesName}`).toggle();
        drawPhenotypePlatesSelection();
    });
}

// draw phenotypes plates selection
function drawPhenotypePlatesSelection() {
    const isNormalized = $('#ckNormalized').is(':checked');
    const selectedPhen = $(`#${selRunPhenotypesName}`).val();
    const selectedNromPhen = $(`#${selRunNormPhenotypesName}`).val();
    const path = isNormalized ? selectedNromPhen : selectedPhen;
    if (!path) { return; }
    projectSelectionStage('Plates');
    console.log(`plates: ${path}`);
    const lockKey = getLock_key();
    GetPhenotypesPlates(path, lockKey, (phenotypePlates) => {
        // plate buttons
        d3.selectAll('.plateSelectionButton').remove();
        const selPlates = d3.select('#divPhenotypePlatesSelecton');
        const buttons = selPlates.selectAll('buttonPlaceholders')
            .data(phenotypePlates)
            .enter()
            .append('a');
        buttons.attr({
            type: 'button',
            class: 'btn btn-default plateSelectionButton',
            id(d) { return `btnPlate${d.index}`; },
            href: '#',
            role: 'button',
        });
        buttons.on('click', (d) => { renderPlate(d); });
        buttons.text(d => `Plate ${d.index + 1}`);
        // griding buton
        $('#divPhenotypePlatesSelecton')
            .append("<a type='button' class='btn btn-default btn-xs plateSelectionButton' id='btnShowGrid' href='#' role='button'>Show Grid</a>");
        $('#btnShowGrid').click(showGrid);
        // check for plate index or load plate 0 by default
        const plateIdx = $('#currentSelection').data('plateIdx');
        let plateId = 'btnPlate0';
        if (plateIdx) plateId = `btnPlate${plateIdx}`;
        document.getElementById(plateId).click();
    });
}

function showGrid() {
    const plateIdx = $('#currentSelection').data('plateIdx');
    const project = $('#currentSelection').data('project');
    const path = `/api/results/gridding/${plateIdx}/${project}`;
    $('#imgGridding').attr('src', baseUrl + path);
    $('#dialogGrid').show();
    $('#dialogGrid').dialog();
}

// draw plate
function renderPlate(phenotypePlates) {
    const path = phenotypePlates.url;
    const plateIdx = phenotypePlates.index;
    const project = $('#spProject').text();
    console.log(`experiment: ${path}`, phenotypePlates);
    $('#currentSelection').data('plateIdx', plateIdx);
    $('#currentSelection').data('project', project);
    $('#spnPlateIdx').text((plateIdx + 1));
    wait();
    // e.g. /api/results/phenotype/GenerationTimeWhen/1/by4742_h/analysis
    const isNormalized = $('#ckNormalized').is(':checked');
    const phenotypePath = isNormalized ? '/api/results/normalized_phenotype/###/' : '/api/results/phenotype/###/';
    const metaDataPath = `${phenotypePath + plateIdx}/${project}`;
    const lockKey = getLock_key();
    GetPlateData(path, isNormalized, metaDataPath, '###', lockKey, (data) => {
        $('#plate').empty();
        const allNull = isPlateAllNull(data.plate_data);
        if (!data || allNull) {
            stopWait();
            return;
        }
        const plateData = data.plate_data;
        const plateMetaData = data.Plate_metadata;
        const growthMetaData = data.Growth_metaData;
        const phenotypeName = data.plate_phenotype;
        qIndexQueue = data.plate_qIdxSort;
        window.qc.actions.retrievePlateCurves(parseInt(plateIdx, 10));
        const plate = DrawPlate('#plate', plateData, growthMetaData, plateMetaData, phenotypeName, dispatch);
        const row = $('#currentSelection').data('row');
        const col = $('#currentSelection').data('col');
        if (row && col) { setExperimentByCoord(row, col); }
        stopWait();
        plate.on('SelectedExperiment', (datah) => {
            const arr = datah.coord.split(',');
            const row = arr[0];
            const col = arr[1];
            $('#currentSelection').data('expId', datah.id);
            $('#currentSelection').data('plateIdx', plateIdx);
            $('#currentSelection').data('row', row);
            $('#currentSelection').data('col', col);
            $('#currentSelection').data('phenotype', datah.phenotype);
            $('#currentSelection').data('project', $('#spProject').text());
            updateQIndexCoord(qIdxOperations.Goto, getQIndexFromCoord(row, col));
            window.qc.actions.setFocus(
                parseInt(plateIdx, 10),
                parseInt(row, 10),
                parseInt(col, 10),
            );
        });
        setExperimentByQidx(qIdxOperations.Reset);
    });
}
