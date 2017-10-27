/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__(1);


/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
    value: true
});

var _helpers = __webpack_require__(2);

var _api = __webpack_require__(3);

window.cccFunctions = {
    setStep: function setStep(step) {
        switch (step) {
            case 0:
                $("#divImageProcessing").hide();
                $("#divProcessImageStep1").hide();
                $("#divProcessImageStep2").hide();
                $("#divProcessImageStep3").hide();
                $("#divIniCCC").hide();
                break;
            case 1:
                $("#divImageProcessing").show();
                $("#divProcessImageStep1").show();
                $("#divProsessImageProgress").hide();
                $("#divPlateSelection").hide();
                $("#imProgress0").hide();
                $("#imProgress1").hide();
                $("#imProgress2").hide();
                $("#imProgress3").hide();
                $("#imProgress4").hide();
                $("#imProgress5").hide();
                $("#imProgress6").hide();
                $("#divGridding").hide();
                break;
            case 1.1:
                $("#divImageSelect").hide();
                $("#imProgress0").show();
                $("#imProgress1").show();
                break;
            case 1.2:
                $("#divProsessImageProgress").show();
                break;
            case 1.3:
                $("#imProgress2").show();
                break;
            case 1.4:
                $("#imProgress3").show();
                break;
            case 1.5:
                $("#imProgress4").show();
                break;
            case 1.6:
                $("#imProgress5").show();
                break;
            case 1.7:
                $("#imProgress6").show();
                break;
            case 1.8:
                $("#divProsessImageProgress").hide();
                $("#divPlateSelection").show();
                break;
            case 2:
                $("#divProcessImageStep1").hide();
                $("#divProcessImageStep2").show();
                $("#divGridding").hide();
                break;
            case 2.1:
                $("#divGridding").show();
                $("#divRegrid").hide();
                $("#divShowColonyDetectionStep").hide();
                break;
            case 2.2:
                $("#cvsPlateGrid").show();
                $("#divShowColonyDetectionStep").show();
                break;
            case 2.3:
                $("#cvsPlateGrid").show();
                $("#divRegrid").show();
                break;
            case 2.4:
                $("#cvsPlateGrid").hide();
                $("#divRegrid").hide();
                $("#divShowColonyDetectionStep").hide();
                break;
            case 3:
                $("#divProcessImageStep1").hide();
                $("#divProcessImageStep2").hide();
                $("#divProcessImageStep3").show();
                $("#divColony").text("Gridding successfull, ploting grid");
                break;
            case 3.1:
                //setColonyUpdateMetaDataButtonsState Ini
                $("#btnFixColonyMetadata").show();
                $("#btnSetUpdatedColonyMetadata").hide();
                $("#btnColonyBlobSizePlus").hide();
                $("#btnColonyBlobSizeMinus").hide();
                $("#btnNextColonyDetect").show();
                break;
            case 3.2:
                //setColonyUpdateMetaDataButtonsState update
                $("#btnFixColonyMetadata").hide();
                $("#btnSetUpdatedColonyMetadata").show();
                $("#btnColonyBlobSizePlus").show();
                $("#btnColonyBlobSizeMinus").show();
                $("#btnNextColonyDetect").hide();
                break;
            default:
        }
    },
    initiateNewCcc: function initiateNewCcc(species, reference, allFields) {
        var valid = true;
        var validNameRegexp = /^[a-z]([0-9a-z_\s])+$/i;
        var invalidNameMsg = "This field may consist of a-z, 0-9, underscores and spaces, and must begin with a letter.";

        allFields.removeClass("ui-state-error");

        valid = valid && cccFunctions.checkLength(species, 3, 20, "species");
        valid = valid && cccFunctions.checkLength(reference, 3, 20, "reference");

        valid = valid && cccFunctions.checkName(species, validNameRegexp, invalidNameMsg);
        valid = valid && cccFunctions.checkName(reference, validNameRegexp, invalidNameMsg);

        if (valid) {
            (0, _api.InitiateCCC)(species.val(), reference.val(), cccFunctions.initiateCccSuccess, cccFunctions.initiateCccError);
        }
        return valid;
    },
    initiateCccError: function initiateCccError(data) {
        cccFunctions.updateTips(data.responseJSON.reason);
    },
    checkLength: function checkLength(obj, min, max, field) {
        if (obj.val().length > max || obj.val().length < min) {
            obj.addClass("ui-state-error");
            cccFunctions.updateTips('Length of ' + field + ' must be between ' + min + ' and ' + max + '.');
            return false;
        } else {
            return true;
        }
    },
    checkName: function checkName(obj, regexp, message) {
        if (!regexp.test(obj.val())) {
            obj.addClass("ui-state-error");
            cccFunctions.updateTips(message);
            return false;
        } else {
            return true;
        }
    }
};

window.executeCCC = function () {

    'use strict';

    var selFixtureName = "selFixtures";
    var selPinFormatsName = "selPinFormats";
    var dialogCCCIni;
    var form;
    var metaDataCanvasState;
    var species = $("#inSpecies");
    var reference = $("#inReference");
    var allFields = $([]).add(species).add(reference);
    var tips = $(".validateTips");

    dialogCCCIni = $("#dialogFormIniCCC").dialog({
        autoOpen: false,
        height: 400,
        width: 350,
        modal: true,
        buttons: {
            "Initiate new CCC": function InitiateNewCCC() {
                return cccFunctions.initiateNewCcc(species, reference, allFields);
            },
            Cancel: function Cancel() {
                dialogCCCIni.dialog("close");
            }
        },
        close: function close() {
            form[0].reset();
            allFields.removeClass("ui-state-error");
        }
    });

    form = dialogCCCIni.find("form").on("submit", function (event) {
        event.preventDefault();
        cccFunctions.initiateNewCcc(species, reference, allFields);
    });

    $("#btnIniCCC").click(openCCCIniDialog);
    $("#btnUploadImage").click(uploadImage);
    $("#btnUploadGridding").click(startGridding);
    $("#btnProcessNewImage").click(initiateProcessImageWizard);
    $("#inImageUpload").change(uploadImage);

    ini();

    function ini() {
        cccFunctions.setStep(0);
        (0, _api.GetFixtures)(function (fixtures) {
            if (fixtures == null) {
                alert("ERROR: There was a problem with the API while fecthing fixtures!");
                return;
            }
            var elementName = selFixtureName;
            d3.select("#" + elementName).remove();
            var selPhen = d3.select("#divFixtureSelector").append("select").attr("id", elementName);
            var options = selPhen.selectAll("optionPlaceholders").data(fixtures).enter().append("option");
            options.attr("value", function (d) {
                return d;
            });
            options.text(function (d) {
                return d;
            });
            $("#" + elementName).selectedIndex = 0;
        });
        (0, _api.GetPinningFormats)(function (formats) {
            if (formats == null) {
                alert("ERROR: There was a problem with the API while fetching pinnnig formats!");
                return;
            }
            var elementName = selPinFormatsName;
            d3.select("#" + elementName).remove();
            var selPhen = d3.select("#divPinningFormatsSelector").append("select").attr("id", elementName);
            var options = selPhen.selectAll("optionPlaceholders").data(formats).enter().append("option");
            options.attr("value", function (d) {
                return d.value;
            });
            options.text(function (d) {
                return d.name;
            });
            $("#" + elementName).selectedIndex = 0;
        });
    };

    function updateTips(t) {
        tips.text(t).addClass("ui-state-highlight");
        setTimeout(function () {
            tips.removeClass("ui-state-highlight", 1500);
        }, 500);
    }

    cccFunctions.updateTips = updateTips;

    function initiateCccSuccess(data) {
        if (data.success) {
            $("#btnIniCCC").hide();
            $("#divIniCCC").show();
            var id = data.identifier;
            var token = data.access_token;
            var sp = species.val();
            var ref = reference.val();
            var fixture = getSelectedFixtureName();
            var pinFormat = getSelectedPinningFormat();
            var outputFormat = getSelectedPinningFormatName();
            $("#tblCurrentCCC tbody").append("<tr><td>Id</td><td>" + id + "</td></tr>" + "<tr><td>Token</td><td>" + token + "</td></tr>" + "<tr><td>Species</td><td>" + sp + "</td></tr>" + "<tr><td>Reference</td><td>" + ref + "</td></tr>" + "<tr><td>Pinning Format</td><td>" + outputFormat + "</td></tr>" + "<tr><td>Fixture</td><td>" + fixture + "</td></tr>" + "<tr><td>Uploaded Images</td><td></td></tr>");
            setCccId(id);
            settAccessToken(token);
            setCccFixture(fixture);
            setCccPinningFormat(pinFormat);
            dialogCCCIni.dialog("close");
        } else {
            alert("Problem initializing:" + data.reason);
        }
    }

    cccFunctions.initiateCccSuccess = initiateCccSuccess;

    function openCCCIniDialog() {
        dialogCCCIni.dialog("open");
    }

    function getSelectedFixtureName() {
        return $("#" + selFixtureName + " option:selected").text();
    }

    function getSelectedPinningFormat() {
        return $("#" + selPinFormatsName + " option:selected").val();
    }

    function getSelectedPinningFormatName() {
        return $("#" + selPinFormatsName + " option:selected").text();
    }

    function getCccId() {
        return $("#inData").data("idCCC");
    }

    function setCccId(id) {
        $("#inData").data("idCCC", id);
    }

    function getAccessToken() {
        return $("#inData").data("accessToken");
    };

    function settAccessToken(token) {
        $("#inData").data("accessToken", token);
    };

    function getCccFixture() {
        return $("#inData").data("idCCCFixture");
    }

    function setCccFixture(name) {
        $("#inData").data("idCCCFixture", name);
    }

    function getCccPinningFormat() {
        return $("#inData").data("idCCCPinningFormat");
    }

    function setCccPinningFormat(name) {
        $("#inData").data("idCCCPinningFormat", name);
    }

    //private functions
    function processMarkers(markers) {
        var markerXcoords = [];
        var markerYcoords = [];
        for (var i = 0; i < markers.length; i++) {
            markerXcoords.push(markers[i][0]);
            markerYcoords.push(markers[i][1]);
        }
        var postMarkers = [];
        postMarkers.push(markerXcoords);
        postMarkers.push(markerYcoords);
        return postMarkers;
    }

    function createFixturePlateSelection(plates, scope) {
        cccFunctions.setStep(1.8);
        d3.selectAll(".frmPlateSeclectionInput").remove();
        var selPlates = d3.select("#frmPlateSeclection");
        var items = selPlates.selectAll("inputPlaceholders").data(plates).enter().append("div");
        items.append("input").attr({
            "type": "checkbox",
            "class": "frmPlateSeclectionInput oneLine rightPadding",
            "id": function id(d) {
                return "inImagePlate" + d.index;
            },
            "name": "ImagePlates",
            "value": function value(d) {
                return d.index;
            }
        });
        items.append("label").text(function (d) {
            return "Plate " + d.index;
        }).classed("oneLine", true);
        setCurrentScope(scope);
    }

    //main functions

    function initiateProcessImageWizard() {
        cccFunctions.setStep(1);
    }

    function uploadImage() {
        var file = $("#inImageUpload")[0].files[0];
        if (!file) {
            alert("You need to select a valid file!");
            return;
        }

        cccFunctions.setStep(1.1);

        $("#tblUploadedImages tbody").append("<tr><td>" + file.name + "</td><td>" + file.size + "</td><td>" + file.type + "</td></tr>");

        var scope = createScope();
        scope.File = file;
        scope.FixtureName = getCccFixture();
        scope.cccId = getCccId();
        scope.AccessToken = getAccessToken();
        var pinFormat = getCccPinningFormat();
        scope.PinFormat = pinFormat.split(",");
        cccFunctions.setStep(1.2);
        (0, _api.GetMarkers)(scope, scope.FixtureName, scope.File, getMarkersSuccess, getMarkersError);
    }

    function getMarkersError(data) {
        alert("Markers error:" + data.responseText);
    }

    function getMarkersSuccess(data, scope) {
        if (data.success == true) {
            cccFunctions.setStep(1.3);
            scope.Markers = processMarkers(data.markers);
            var file = scope.File;
            scope.File = null;
            (0, _api.GetImageId)(scope, scope.cccId, file, scope.AccessToken, getImageIdSuccess, getImageIdError);
        } else alert("There was a problem with the upload: " + data.reason);
    }

    function getImageIdError(data) {
        alert("Fatal error uploading the image: \n " + data.responseText);
    }

    function getImageIdSuccess(data, scope) {
        if (data.success) {
            cccFunctions.setStep(1.4);
            scope.CurrentImageId = data.image_identifier;
            var markers = scope.Markers;
            var toSetData = [];
            toSetData.push({ key: "marker_x", value: markers[0] });
            toSetData.push({ key: "marker_y", value: markers[1] });
            (0, _api.SetCccImageData)(scope, scope.cccId, scope.CurrentImageId, scope.AccessToken, toSetData, scope.FixtureName, setCccImageDataSuccess, setCccImageDataError);
        } else alert("there was a problem uploading the image:" + data.reason);
    }

    function setCccImageDataError(data) {
        alert("Error while setting up the images! " + data.responseText);
    }

    function setCccImageDataSuccess(data, scope) {
        if (data.success) {
            cccFunctions.setStep(1.5);
            (0, _api.SetCccImageSlice)(scope, scope.cccId, scope.CurrentImageId, scope.AccessToken, setCccImageSliceSuccess, setCccImageSliceError);
        } else alert("set image error!");
    }

    function setCccImageSliceError(data) {
        alert("Error while setting up the images slice!" + data.responseText);
    }

    function setCccImageSliceSuccess(data, scope) {
        if (data.success) {
            cccFunctions.setStep(1.6);
            (0, _api.SetGrayScaleImageAnalysis)(scope, scope.cccId, scope.CurrentImageId, scope.AccessToken, setGrayScaleImageAnalysisSuccess, setGrayScaleImageAnalysisError);
        } else alert("Error while setting up the images slice!:" + data.reason);
    }

    function setGrayScaleImageAnalysisError(data) {
        alert("Error while starting grayscale analysis! " + data.responseText);
    }

    function setGrayScaleImageAnalysisSuccess(data, scope) {
        if (data.success) {
            cccFunctions.setStep(1.7);
            //store target_values and source_values to QC graph ???
            (0, _api.GetFixturePlates)(scope.FixtureName, function (data) {
                createFixturePlateSelection(data, scope);
            });
        } else alert("Error while starting grayscale analysis:" + data.reason);
    }

    function startGridding() {
        var cbs = document.forms["frmPlateSeclection"].elements["ImagePlates"];
        var plateId;
        var task;
        var scope;
        if (cbs.constructor === Array) {
            for (var i = 0, cbLen = cbs.length; i < cbLen; i++) {
                if (cbs[i].checked) {
                    plateId = cbs[i].value;
                    scope = getCurrentScope();
                    task = createSetGrayScaleTransformTask(scope, plateId);
                    $(document).queue("plateProcess", task);
                }
            }
        } else {
            if (cbs.checked) {
                plateId = cbs.value;
                scope = getCurrentScope();
                task = createSetGrayScaleTransformTask(scope, plateId);
                $(document).queue("plateProcess", task);
            } else alert("No plate was selected!");
        }
        $(document).dequeue("plateProcess");
    };

    function createSetGrayScaleTransformTask(scope, plate) {
        return function (next) {
            cccFunctions.setStep(2);
            scope.Plate = plate;
            scope.PlateNextTaskInQueue = next;
            (0, _api.SetGrayScaleTransform)(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken, setGrayScaleTransformSuccess, setGrayScaleTransformError);
        };
    }

    cccFunctions.createSetGrayScaleTransformTask = createSetGrayScaleTransformTask;

    function setGrayScaleTransformError(data) {
        alert("set grayscale transform error:" + data.reason);
    }

    cccFunctions.setGrayScaleTransformError = setGrayScaleTransformError;

    function setGrayScaleTransformSuccess(data, scope) {
        if (data.success) {
            cccFunctions.setStep(2.1);
            $("#divGridStatus").text("Calculating Gridding ... please wait ...!");
            (0, _api.SetGridding)(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.PinFormat, [0, 0], scope.AccessToken, setGriddingSuccess, setGriddingError);
        } else alert("set grayscale transform error:" + data.reason);
    }

    cccFunctions.setGrayScaleTransformSuccess = setGrayScaleTransformSuccess;

    function drawCanvasGridCentersReGrid(scope, ctx) {
        var first = true;
        var size = 20;
        var totalRows = scope.PlateGridding.grid[0].length;
        var totalCols = scope.PlateGridding.grid[0][0].length;
        for (var row = 0; row < totalRows; row++) {
            for (var col = 0; col < totalCols; col++) {
                var x = scope.gridding.grid[1][row][col];
                var y = scope.gridding.grid[0][row][col];
                ctx.beginPath();
                ctx.arc(x, y, size, 0, 2 * Math.PI);
                ctx.lineWidth = 3;
                ctx.strokeStyle = "red";
                ctx.closePath();
                if (first) {
                    ctx.fillStyle = "#c82124";
                    ctx.fill();
                    first = false;
                }
                ctx.stroke();
            }
        }
    }

    function drawCanvasGridCenters(scope, ctx, currentRow, currentCol) {

        var markCurrentPos = false;
        if (currentRow != undefined && currentCol != undefined) markCurrentPos = true;
        var first = true;
        var outlineSize = 20;
        var markerSize = 40;
        var totalRows = scope.PlateGridding.grid[0].length;
        var totalCols = scope.PlateGridding.grid[0][0].length;
        for (var row = 0; row < totalRows; row++) {
            for (var col = 0; col < totalCols; col++) {
                renderColonyOutlines(row, col, outlineSize);
                renderColonyMarker(row, col, currentRow, currentCol, markerSize);
            }
        }

        function renderColonyOutlines(row, col, size) {
            var x = scope.PlateGridding.grid[1][row][col];
            var y = scope.PlateGridding.grid[0][row][col];
            if (first) {
                ctx.fillStyle = "#c82124";
            }
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI);
            if (first) {
                ctx.closePath();
                ctx.fill();
                first = false;
            } else ctx.stroke();
        }

        function renderColonyMarker(row, col, currentRow, currentCol, size) {
            if (markCurrentPos && col === currentCol && row === currentRow) {
                var x = scope.PlateGridding.grid[1][row][col];
                var y = scope.PlateGridding.grid[0][row][col];
                ctx.beginPath();
                ctx.arc(x, y, size, 0, 2 * Math.PI);
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
    }

    function drawCanvasGrid(scope, ctx) {
        var totalRows = scope.gridding.xy1.length;
        var totalCols = scope.gridding.xy1[0].length;
        for (var row = 0; row < totalRows; row++) {
            for (var col = 0; col < totalCols; col++) {
                var p1 = scope.gridding.xy1[row][col];
                var p2 = scope.gridding.xy2[row][col];
                var x1 = p1[1];
                var y1 = p1[0];
                var x2 = p2[1];
                var y2 = p2[0];
                var w = x2 - x1;
                var h = y2 - y1;
                ctx.rect(x1, y1, w, h);
                ctx.stroke();
            }
        }
    }

    function renderGridFail(data, scope) {
        scope.PlateGridding = {
            grid: data.grid
        };
        var imgPlateSlice = new Image();
        imgPlateSlice.src = (0, _api.GetSliceImageURL)(scope.cccId, scope.CurrentImageId, scope.Plate);
        $("#divGridStatus").text($("#divGridStatus").text() + "\nFecthing Plate Slice ...");

        imgPlateSlice.onload = function () {
            $("#divGridStatus").text($("#divGridStatus").text() + "\nDrawing gridding...");
            plotGridOnPlateSliceInCanvas(scope, imgPlateSlice, "cvsPlateGrid");
            //cccFunctions.setStep(3.1);
            $("#btnReGrid").click(function () {
                cccFunctions.setStep(2.4);
                var offsetRow = $("#inOffsetRow").val();
                var offsetCol = $("#inOffsetCol").val();
                var offset = [];
                offset.push(offsetRow);
                offset.push(offsetCol);
                $("#divGridStatus").text("Calculating Gridding ... please wait ...!");
                (0, _api.SetGridding)(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.PinFormat, offset, scope.AccessToken, setGriddingSuccess, setGriddingError);
            });
        };
    }

    cccFunctions.renderGridFail = renderGridFail;

    function renderGrid(data, scope) {
        scope.PlateGridding = {
            grid: data.grid,
            xy1: data.xy1,
            xy2: data.xy2
        };

        var imgPlateSlice = new Image();
        var sliceUrl = (0, _api.GetSliceImageURL)(scope.cccId, scope.CurrentImageId, scope.Plate);
        imgPlateSlice.src = sliceUrl;
        (0, _helpers.getDataUrlfromUrl)(sliceUrl, function (dataUrl) {
            scope.PlateDataURL = dataUrl;
        });
        imgPlateSlice.onload = function () {
            plotGridOnPlateSliceInCanvas(scope, imgPlateSlice, "cvsPlateGrid");
            $("#btnStartColonyDetection").click(function () {
                cccFunctions.setStep(3);
                iniColonyStats(scope);
                plotGridOnPlateSliceInCanvas(scope, imgPlateSlice, "cvsPlateGridColonyMarker", 0, 0);
                scope.PlateCurrentColonyRow = 0;
                scope.PlateCurrentColonyCol = 0;
                detectColony(scope, 0, 0);
            });
        };
    }

    cccFunctions.renderGrid = renderGrid;

    function nextColonyDetection(scope) {
        //var scope = getCurrentScope();
        var plateRows = scope.PinFormat[1];
        var plateCols = scope.PinFormat[0];
        var colonyRow = scope.PlateCurrentColonyRow;
        var colonyCol = scope.PlateCurrentColonyCol;

        colonyCol += 1;
        if (colonyCol >= plateCols) {
            colonyRow += 1;
            colonyCol = 0;
        }

        $("#divColony").text("Colony: " + colonyRow + "," + colonyCol);
        moveProgress(colonyRow, colonyCol, plateRows, plateCols);

        var img = new Image();
        img.src = scope.PlateDataURL;
        img.onload = function () {
            plotGridOnPlateSliceInCanvas(scope, img, "cvsPlateGridColonyMarker", colonyRow, colonyCol);
            detectColony(scope, colonyRow, colonyCol);
        };
    }

    function plotGridOnPlateSliceInCanvas(scope, image, canvasId, row, col) {
        var scaleFactor = 0.25;

        var canvas = document.getElementById(canvasId);
        canvas.width = image.naturalWidth * scaleFactor;
        canvas.height = image.naturalHeight * scaleFactor;
        var ctx = canvas.getContext('2d');
        ctx.scale(0.2, 0.2);

        ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);
        drawCanvasGridCenters(scope, ctx, row, col);
        //optional squares
        //drawCanvasGrid(scope, ctx);
    }

    function setGriddingError(data, scope) {
        cccFunctions.setStep(2.3);
        $("#divGridStatus").text("Gridding was unsuccesful. Reason: '" + data.reason + "'. Please enter Offset and retry!");
        cccFunctions.renderGridFail(data, scope);
    }

    cccFunctions.setGriddingError = setGriddingError;

    function setGriddingSuccess(data, scope) {
        cccFunctions.setStep(2.2);
        $("#divGridStatus").text("Gridding was sucessful!");
        cccFunctions.renderGrid(data, scope);
    }

    cccFunctions.setGriddingSuccess = setGriddingSuccess;

    function doDetectColonyTask(scope, row, col, next) {
        scope.PlateColonyNextTaskInQueue = next;
        detectColony(scope, row, col);
    }

    function detectColony(scope, row, col) {
        scope.PlateCurrentColonyRow = row;
        scope.PlateCurrentColonyCol = col;
        (0, _api.SetColonyDetection)(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken, row, col, setColonyDetectionSuccess, setColonyDetectionError);
    }

    function createDetectColonyTask(scope, row, col) {
        return function (next) {
            doDetectColonyTask(scope, row, col, next);
        };
    }

    function moveProgress(row, col, totalRow, totalCol) {
        var m = row * totalCol + col;
        var percent = m * 100 / (totalCol * totalRow);
        $("#progressbar").progressbar({ value: percent });
        $("#divPro").text("col:" + col + " row:" + row + " m: " + m + " %:" + percent);
    }

    function setColonyDetectionSuccess(data, scope, row, col) {
        if (data.success) {
            var colony = {
                x: row,
                y: col,
                blob: data.blob,
                blobMax: data.blob_max,
                blobMin: data.blob_min,
                blobExist: data.blob_exists,
                background: data.background,
                backgroundExists: data.background_exists,
                backgroundReasonable: data.background_reasonable,
                image: data.image,
                imageMax: data.image_max,
                imageMin: data.image_min,
                gridPosition: data.grid_position
            };
            scope.PlateCurrentColony = colony;
            cccFunctions.setStep(3.1);

            var canvasBg = document.getElementById("colonyImageCanvas");
            var canvasFg = document.getElementById("colonyMarkingsCanvas");
            var canvasMarkings = document.getElementById("colonyPlotCanvas");

            (0, _helpers.createCanvasMarker)(colony, canvasMarkings);
            (0, _helpers.createCanvasImage)(colony, canvasBg);

            canvasFg.width = canvasBg.width;
            canvasFg.height = canvasBg.height;

            setCurrentScope(scope);
            //runNextPlateColonyTask(scope);
        } else {
            alert("no success");
        }
    }

    $("#btnFixColonyMetadata").click(function () {
        var canvasBg = document.getElementById("colonyImageCanvas");
        var canvasFg = document.getElementById("colonyMarkingsCanvas");
        metaDataCanvasState = new CanvasState(canvasFg);

        cccFunctions.setStep(3.2);
        metaDataCanvasState.addShape(new Blob(canvasBg.width / 2, canvasBg.height / 2, 15, "rgba(255, 0, 0, .2)"));
    });

    function updateColonyData(blob, background, updateBlob, updatedBackground) {
        for (var row = 0; row < blob.length; row++) {
            for (var col = 0; col < blob[0].length; col++) {
                var valueBlob = blob[row][col];
                var valuebackground = background[row][col];
                if (!valueBlob && !valuebackground) continue;
                blob[row][col] = updateBlob[row][col];
                background[row][col] = updatedBackground[row][col];
            }
        }
        var newMetadata = {
            blob: blob,
            background: background
        };
        return newMetadata;
    }

    $("#btnSetUpdatedColonyMetadata").click(function () {
        var canvasMarkings = document.getElementById("colonyPlotCanvas");
        var newMetaData = (0, _helpers.getMarkerData)("colonyMarkingsCanvas");
        var scope = getCurrentScope();

        var updatedMetaData = updateColonyData(scope.PlateCurrentColony.blob, scope.PlateCurrentColony.background, newMetaData.blob, newMetaData.background);
        scope.PlateCurrentColony.blob = updatedMetaData.blob;
        scope.PlateCurrentColony.background = updatedMetaData.background;

        (0, _helpers.createCanvasMarker)(scope.PlateCurrentColony, canvasMarkings);
        cccFunctions.setStep(3.1);
        metaDataCanvasState.shapes = [];
        metaDataCanvasState.needsRender = true;
        metaDataCanvasState = null;

        setCurrentScope(scope);
    });
    $("#btnColonyBlobSizePlus").click(function () {
        var blob = metaDataCanvasState.shapes[0];
        blob.r += 5;
        metaDataCanvasState.needsRender = true;
    });
    $("#btnColonyBlobSizeMinus").click(function () {
        var blob = metaDataCanvasState.shapes[0];
        blob.r -= 5;
        metaDataCanvasState.needsRender = true;
    });
    $("#btnNextColonyDetect").click(function () {
        var scope = getCurrentScope();
        (0, _api.SetColonyCompressionV2)(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken, scope.PlateCurrentColony, scope.PlateCurrentColonyRow, scope.PlateCurrentColonyCol, setColonyCompressionSuccess, setColonyCompressionError);
    });

    function setColonyDetectionError(data) {
        alert("set Colony Detection Error:" + data.reason);
    }

    function setColonyCompressionSuccess(data, scope) {
        nextColonyDetection(scope);
    }

    function setColonyCompressionError(data) {
        alert("set Colony compression Error:" + data.reason);
    }

    function reportColonyDetectionProgress(data, scope) {
        var success = data.success;
        var reason = data.reason;
        if (typeof reason == 'undefined') reason = '';
        var item = success + "- " + reason;
        scope.ColonyDetection.push(item);
        if (success) scope.ColonyDetected = scope.ColonyDetected + 1;else scope.ColonyNotDetected = scope.ColonyNotDetected + 1;
        $("#divCompress").text("Detected:" + scope.ColonyDetected + " fails:" + scope.ColonyNotDetected + " last error:" + data.reason);
    }

    function reportColonyCompressionProgress(data) {
        var success = data.success;
        var reason = data.reason;
        if (typeof reason == 'undefined') reason = '';
        var item = success + "- " + reason;
        scope.ColonyCompression.push(item);
        if (success) scope.ColonyCompressed = scope.ColonyCompressed + 1;else scope.ColonyNotCompressed = scope.ColonyNotCompressed + 1;
        $("#divCompress").text("compressed:" + scope.ColonyCompressed + " fails:" + scope.ColonyNotCompressed + " last error:" + data.reason);
    }

    function runNextPlateColonyTask(scope) {
        var next = scope.PlateColonyNextTaskInQueue;
        if (next == null) {
            alert("the end");
        }
        setTimeout(function () {
            next();
        }, 500);
    }
};

$(document).ready(executeCCC);

exports.default = cccFunctions;

/***/ }),
/* 2 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.createCanvasImage = createCanvasImage;
exports.createCanvasMarker = createCanvasMarker;
exports.getMarkerData = getMarkerData;
exports.hexToRgb = hexToRgb;
exports.getLinearMapping = getLinearMapping;
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
    for (var i = 0; i < imgdatalen; i += 4) {
        //iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        var color = data.image[imageRow][imageCol];
        var mappedColor = cs(color);

        var rgb = hexToRgb(mappedColor);

        imgdata.data[i + 0] = rgb.r; // RED (0-255)
        imgdata.data[i + 1] = rgb.g; // GREEN (0-255)
        imgdata.data[i + 2] = rgb.b; // BLUE (0-255)
        imgdata.data[i + 3] = 255; // APLHA (0-255)
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
    for (var i = 0; i < imgdatalen; i += 4) {
        //iterate over every pixel in the canvas
        imageCol += 1;
        if (imageCol >= cols) {
            imageCol = 0;
            imageRow += 1;
        }
        var rgb;
        if (data.blob[imageRow][imageCol] === true) {
            rgb = { r: 255, g: 0, b: 0 };
        } else if (data.background[imageRow][imageCol] === true) {
            rgb = { r: 0, g: 128, b: 0 };
        } else {
            rgb = { r: 0, g: 0, b: 0 };
        }

        imgdata.data[i + 0] = rgb.r; // RED (0-255)
        imgdata.data[i + 1] = rgb.g; // GREEN (0-255)
        imgdata.data[i + 2] = rgb.b; // BLUE (0-255)
        imgdata.data[i + 3] = 255; // APLHA (0-255)
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

    for (var i = 0; i < imgdatalen; i += 4) {
        //iterate over every pixel in the canvas
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

    var cs = d3.scale.linear().domain([intensityMin, intensityMean, intensityMax]).range([colorScheme[2], colorScheme[1], colorScheme[0]]);

    return cs;
}

/***/ }),
/* 3 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.GetSliceImageURL = GetSliceImageURL;
exports.GetSliceImage = GetSliceImage;
exports.GetFixtures = GetFixtures;
exports.GetFixtureData = GetFixtureData;
exports.GetFixturePlates = GetFixturePlates;
exports.GetPinningFormats = GetPinningFormats;
exports.GetPinningFormatsv2 = GetPinningFormatsv2;
exports.InitiateCCC = InitiateCCC;
exports.SetCccImageData = SetCccImageData;
exports.SetCccImageSlice = SetCccImageSlice;
exports.SetGrayScaleImageAnalysis = SetGrayScaleImageAnalysis;
exports.GetGrayScaleAnalysis = GetGrayScaleAnalysis;
exports.SetGrayScaleTransform = SetGrayScaleTransform;
exports.SetGridding = SetGridding;
exports.SetColonyDetection = SetColonyDetection;
exports.SetColonyCompression = SetColonyCompression;
exports.SetColonyCompressionV2 = SetColonyCompressionV2;
exports.GetImageId = GetImageId;
exports.GetMarkers = GetMarkers;
exports.GetTransposedMarkersV2 = GetTransposedMarkersV2;
exports.GetTransposedMarkers = GetTransposedMarkers;
exports.GetAPILock = GetAPILock;
exports.addCacheBuster = addCacheBuster;
exports.addKeyParameter = addKeyParameter;
exports.RemoveLock = RemoveLock;
//var baseUrl = "http://localhost:5000";
var baseUrl = "";
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

    d3.json(path, function (error, json) {
        if (error) console.warn(error);else {
            var fixtrues = json.fixtures;
            callback(fixtrues);
        }
    });
};

function GetFixtureData(fixtureName, callback) {
    var path = GetFixtruesDataPath + fixtureName;

    d3.json(path, function (error, json) {
        if (error) console.warn(error);else {
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
        if (error) console.warn(error);else {
            var fixtrues = json.pinning_formats;
            callback(fixtrues);
        }
    });
};

function GetPinningFormatsv2(successCallback, errorCallback) {
    var path = GetPinningFormatsPath;

    $.ajax({
        url: path,
        type: "GET",
        success: successCallback,
        error: errorCallback
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

function SetCccImageData(scope, cccId, imageId, accessToken, dataArray, fixture, successCallback, errorCallback) {
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
        success: function success(data) {
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function SetCccImageSlice(scope, cccId, imageId, accessToken, successCallback, errorCallback) {
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
        success: function success(data) {
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function SetGrayScaleImageAnalysis(scope, cccId, imageId, accessToken, successCallback, errorCallback) {
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
        success: function success(data) {
            successCallback(data, scope);
        },
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
        success: function success(data) {
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function SetGridding(scope, cccId, imageId, plate, pinningFormat, offSet, accessToken, successCallback, errorCallback) {
    var path = SetGriddingPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate);
    $.ajax({
        url: path,
        type: "POST",
        dataType: 'json',
        contentType: 'application/json',
        data: JSON.stringify({
            pinning_format: pinningFormat,
            gridding_correction: offSet,
            access_token: accessToken
        })
    }).done(function (data) {
        successCallback(data, scope);
    }).fail(function (jqXHR) {
        var data = JSON.parse(jqXHR.responseText);
        errorCallback(data, scope);
    });
}

function SetColonyDetection(scope, cccId, imageId, plate, accessToken, row, col, successCallback, errorCallback) {
    var path = SetColonyDetectionPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate).replace("#3#", col).replace("#4#", row);

    var formData = new FormData();
    formData.append("access_token", accessToken);
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function success(data) {
            successCallback(data, scope, row, col);
        },
        error: errorCallback
    });
}

function SetColonyCompression(scope, cccId, imageId, plate, accessToken, colony, row, col, successCallback, errorCallback) {
    var path = SetColonyCompressionPath.replace("#0#", cccId).replace("#1#", imageId).replace("#2#", plate).replace("#3#", row).replace("#4#", col);

    var formData = new FormData();
    formData.append("access_token", accessToken);
    formData.append("image", JSON.stringify(colony.image));
    formData.append("blob", JSON.stringify(colony.blob));
    formData.append("background", JSON.stringify(colony.background));
    $.ajax({
        url: path,
        type: "POST",
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function success(data) {
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
        success: function success(data) {
            scope.ColonyRow = row;
            scope.ColonyCol = col;
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function GetImageId(scope, cccId, file, accessToken, successCallback, errorCallback) {
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
        success: function success(data) {
            successCallback(data, scope);
        },
        error: errorCallback
    });
}

function GetMarkers(scope, fixtureName, file, successCallback, errorCallback) {
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
        success: function success(data) {
            successCallback(data, scope);
        },
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
        var path = baseUrl + url + "/" + addCacheBuster(true);
        d3.json(path, function (error, json) {
            if (error) return console.warn(error);else {
                var permissionText;
                var lock;
                if (json.success == true) {
                    lock = json.lock_key;
                    permissionText = "Read/Write";
                } else {
                    permissionText = "Read Only";
                    lock = null;
                }
                var lockData = {
                    lock_key: lock,
                    lock_state: permissionText
                };
                callback(lockData);
            }
        });
    }
};

function addCacheBuster(fisrt) {
    if (fisrt === true) return "?buster=" + Math.random().toString(36).substring(7);
    return "&buster=" + Math.random().toString(36).substring(7);
}

function addKeyParameter(key) {
    if (key) return "?lock_key=" + key + addCacheBuster();else return "";
}

function RemoveLock(url, key, callback) {
    var path = baseUrl + url + addKeyParameter(key);

    d3.json(path, function (error, json) {
        if (error) return console.warn(error);else {
            callback(json.success);
        }
    });
};

/***/ })
/******/ ]);