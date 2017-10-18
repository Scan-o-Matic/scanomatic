$(document)
    .ready(function () {

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
                "Initiate new CCC": initiateNewCcc,
                Cancel: function() { dialogCCCIni.dialog("close"); }
            },
            close: function() {
                form[0].reset();
                allFields.removeClass("ui-state-error");
            }
        });

        form = dialogCCCIni.find("form").on("submit",
            function(event) {
                event.preventDefault();
                initiateNewCcc();
            });

        $("#btnIniCCC").click(openCCCIniDialog);
        $("#btnUploadImage").click(uploadImage);
        $("#btnUploadGridding").click(startGridding);
        $("#btnProcessNewImage").click(initiateProcessImageWizard);
        $("#inImageUpload").change(uploadImage);


        ini();

        function ini() {
            setStep(0);
            GetFixtures(function(fixtures) {
                if (fixtures == null) {
                    alert("ERROR: There was a problem with the API while fecthing fixtures!");
                    return;
                }
                var elementName = selFixtureName;
                d3.select("#" + elementName).remove();
                var selPhen = d3.select("#divFixtureSelector")
                    .append("select")
                    .attr("id", elementName);
                var options = selPhen.selectAll("optionPlaceholders")
                    .data(fixtures)
                    .enter()
                    .append("option");
                options.attr("value", function(d) { return d; });
                options.text(function(d) { return d; });
                $("#" + elementName).selectedIndex = 0;
            });
            GetPinningFormats(function(formats) {
                if (formats == null) {
                    alert("ERROR: There was a problem with the API while fetching pinnnig formats! ");
                    return;
                }
                var elementName = selPinFormatsName;
                d3.select("#" + elementName).remove();
                var selPhen = d3.select("#divPinningFormatsSelector")
                    .append("select")
                    .attr("id", elementName);
                var options = selPhen.selectAll("optionPlaceholders")
                    .data(formats)
                    .enter()
                    .append("option");
                options.attr("value", function(d) { return d.value; });
                options.text(function(d) { return d.name; });
                $("#" + elementName).selectedIndex = 0;
            });
        };

        function updateTips(t) {
            tips.text(t).addClass("ui-state-highlight");
            setTimeout(function() {
                    tips.removeClass("ui-state-highlight", 1500);
                },
                500);
        }

        function checkLength(o, n, min, max) {
            if (o.val().length > max || o.val().length < min) {
                o.addClass("ui-state-error");
                updateTips("Length of " +
                    n +
                    " must be between " +
                    min +
                    " and " +
                    max +
                    ".");
                return false;
            } else {
                return true;
            }
        }

        function checkRegexp(o, regexp, n) {
            if (!(regexp.test(o.val()))) {
                o.addClass("ui-state-error");
                updateTips(n);
                return false;
            } else {
                return true;
            }
        };

        function openCCCIniDialog() {
            dialogCCCIni.dialog("open");
        }

        function getSelectedFixtureName() {
            var fixture = $("#" + selFixtureName + " option:selected").text();
            return fixture;
        }

        function getSelectedPinningFormat() {
            var format = $("#" + selPinFormatsName + " option:selected").val();
            return format;
        }

        function getSelectedPinningFormatName() {
            var format = $("#" + selPinFormatsName + " option:selected").text();
            return format;
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
            setStep(1.8);
            d3.selectAll(".frmPlateSeclectionInput").remove();
            var selPlates = d3.select("#frmPlateSeclection");
            var items = selPlates.selectAll("inputPlaceholders")
                .data(plates)
                .enter().append("div");
            items.append("input")
                .attr({
                    "type": "checkbox",
                    "class": "frmPlateSeclectionInput oneLine rightPadding",
                    "id": function(d) { return "inImagePlate" + d.index; },
                    "name": "ImagePlates",
                    "value": function(d) { return d.index; }
                });
            items.append("label")
                .text(function(d) { return "Plate " + d.index })
                .classed("oneLine", true);
            setCurrentScope(scope);
        }

        //main functions

        function initiateNewCcc() {
            var valid = true;
            allFields.removeClass("ui-state-error");

            valid = valid && checkLength(species, "species", 3, 20);
            valid = valid && checkLength(reference, "reference", 3, 20);

            valid = valid && checkRegexp(species, /^[a-z]([0-9a-z_\s])+$/i, "This field may consist of a-z, 0-9, underscores, spaces and must begin with a letter.");
            valid = valid && checkRegexp(reference, /^[a-z]([0-9a-z_\s])+$/i, "This field may consist of a-z, 0-9, underscores, spaces and must begin with a letter.");

            if (valid) {
                var sp = species.val();
                var ref = reference.val();
                InitiateCCC(sp, ref, initiateCccSuccess, initiateCccError);
            }
            return valid;
        }

        function initiateCccError(data) {
            alert("ini failure!");
        }

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
                $("#tblCurrentCCC tbody").append(
                    "<tr><td>Id</td><td>" + id + "</td></tr>" +
                    "<tr><td>Token</td><td>" + token + "</td></tr>" +
                    "<tr><td>Species</td><td>" + sp + "</td></tr>" +
                    "<tr><td>Reference</td><td>" + ref + "</td></tr>" +
                    "<tr><td>Pinning Format</td><td>" + outputFormat + "</td></tr>" +
                    "<tr><td>Fixture</td><td>" + fixture + "</td></tr>" +
                    "<tr><td>Uploaded Images</td><td></td></tr>"
                    );
                setCccId(id);
                settAccessToken(token);
                setCccFixture(fixture);
                setCccPinningFormat(pinFormat);
                dialogCCCIni.dialog("close");
            } else {
                alert("Problem initializing:" + data.reason);
            }
        }

        function initiateProcessImageWizard() {
            setStep(1);
        }

        function setStep(step) {
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
        }

        function uploadImage() {
            var file = $("#inImageUpload")[0].files[0];
            if (!file) {
                alert("You need to select a valid file!");
                return;
            }

            setStep(1.1);

            $("#tblUploadedImages tbody").append(
                "<tr><td>" + file.name + "</td><td>" + file.size + "</td><td>" + file.type +"</td></tr>"
            );

            var scope = createScope();
            scope.File = file;
            scope.FixtureName = getCccFixture();
            scope.cccId = getCccId();
            scope.AccessToken = getAccessToken();
            var pinFormat = getCccPinningFormat();
            scope.PinFormat = pinFormat.split(",");
            setStep(1.2);
            GetMarkers(scope, scope.FixtureName, scope.File, getMarkersSuccess, getMarkersError);
        }

        function getMarkersError(data) {
            alert("Markers error:" + data.responseText);
        }

        function getMarkersSuccess(data, scope) {
            if (data.success == true) {
                setStep(1.3);
                scope.Markers = processMarkers(data.markers);
                var file = scope.File;
                scope.File = null;
                GetImageId(scope, scope.cccId, file, scope.AccessToken, getImageIdSuccess, getImageIdError);
            } else
                alert("There was a problem with the upload: " + data.reason);
        }

        function getImageIdError(data) {
            alert("Fatal error uploading the image: \n " + data.responseText);
        }

        function getImageIdSuccess(data, scope) {
            if (data.success) {
                setStep(1.4);
                scope.CurrentImageId = data.image_identifier;
                var markers = scope.Markers;
                var toSetData = [];
                toSetData.push({ key: "marker_x", value: markers[0] });
                toSetData.push({ key: "marker_y", value: markers[1] });
                SetCccImageData(scope, scope.cccId, scope.CurrentImageId, scope.AccessToken, toSetData, scope.FixtureName, setCccImageDataSuccess, setCccImageDataError);
            }
            else
                alert("there was a problem uploading the image:" + data.reason);
        }

        function setCccImageDataError(data) {
            alert("Error while setting up the images! " + data.responseText);
        }

        function setCccImageDataSuccess(data, scope) {
            if (data.success) {
                setStep(1.5);
                SetCccImageSlice(scope, scope.cccId, scope.CurrentImageId, scope.AccessToken, setCccImageSliceSuccess, setCccImageSliceError);
            } else
                alert("set image error!");
        }

        function setCccImageSliceError(data) {
            alert("Error while setting up the images slice!" + data.responseText);
        }

        function setCccImageSliceSuccess(data, scope) {
            if (data.success) {
                setStep(1.6);
                SetGrayScaleImageAnalysis(scope, scope.cccId, scope.CurrentImageId, scope.AccessToken, setGrayScaleImageAnalysisSuccess, setGrayScaleImageAnalysisError);
            } else
                alert("Error while setting up the images slice!:" + data.reason);
        }

        function setGrayScaleImageAnalysisError(data) {
            alert("Error while starting grayscale analysis! " + data.responseText);
        }

        function setGrayScaleImageAnalysisSuccess(data, scope) {
            if (data.success) {
                setStep(1.7);
                //store target_values and source_values to QC graph ???
                GetFixturePlates(scope.FixtureName, function(data) {
                    createFixturePlateSelection(data, scope);
                });

            } else
                alert("Error while starting grayscale analysis:" + data.reason);
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
                } else
                    alert("No plate was selected!");
            }
            $(document).dequeue("plateProcess");
        };

        function createSetGrayScaleTransformTask(scope, plate) {
            return function(next) {
                setStep(2);
                scope.Plate = plate;
                scope.PlateNextTaskInQueue = next;
                SetGrayScaleTransform(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken, setGrayScaleTransformSuccess, setGrayScaleTransformError);
            };
        }


        function setGrayScaleTransformError(data) {
            alert("set grayscale transform error:" + data.reason);
        }

        function setGrayScaleTransformSuccess(data, scope) {
            if (data.success) {
                setStep(2.1);
                $("#divGridStatus").text("Calculating Gridding ... please wait ...!");
                SetGridding(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.PinFormat, [0, 0], scope.AccessToken, setGriddingSuccess, setGriddingError);
            } else
                alert("set grayscale transform error:" + data.reason);
        }

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
            if (currentRow != undefined && currentCol != undefined)
                markCurrentPos = true;
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
                }
                else
                    ctx.stroke();
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
            var imgPlateSlice = new Image;
            imgPlateSlice.src = GetSliceImageURL(scope.cccId, scope.CurrentImageId, scope.Plate);
            $("#divGridStatus").text($("#divGridStatus").text()+"\nFecthing Plate Slice ...");

            imgPlateSlice.onload = function () {
                $("#divGridStatus").text($("#divGridStatus").text() +"\nDrawing gridding...");
                plotGridOnPlateSliceInCanvas(scope, imgPlateSlice, "cvsPlateGrid");
                //setStep(3.1);
                $("#btnReGrid").click(function () {
                    setStep(2.4);
                    var offsetRow = $("#inOffsetRow").val();
                    var offsetCol = $("#inOffsetCol").val();
                    var offset = [];
                    offset.push(offsetRow);
                    offset.push(offsetCol);
                    $("#divGridStatus").text("Calculating Gridding ... please wait ...!");
                    SetGridding(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.PinFormat, offset, scope.AccessToken, setGriddingSuccess, setGriddingError);
                });
            }
        }

        function renderGrid(data, scope) {
            scope.PlateGridding = {
                grid: data.grid,
                xy1: data.xy1,
                xy2: data.xy2
            };

            var imgPlateSlice = new Image;
            var sliceUrl = GetSliceImageURL(scope.cccId, scope.CurrentImageId, scope.Plate);
            imgPlateSlice.src = sliceUrl;
            getDataUrlfromUrl(sliceUrl, function (dataUrl) { scope.PlateDataURL = dataUrl; });
            imgPlateSlice.onload = function () {
                plotGridOnPlateSliceInCanvas(scope, imgPlateSlice, "cvsPlateGrid");
                $("#btnStartColonyDetection").click(function () {
                    setStep(3);
                    iniColonyStats(scope);
                    plotGridOnPlateSliceInCanvas(scope, imgPlateSlice, "cvsPlateGridColonyMarker", 0, 0);
                    scope.PlateCurrentColonyRow = 0;
                    scope.PlateCurrentColonyCol = 0;
                    detectColony(scope, 0, 0);
                });
            }
        }

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
            }
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
            setStep(2.3);
            $("#divGridStatus").text("Gridding was unsuccesful. <br>Reason: " + data.reason + "<br>Please enter Offset and retry!");
            renderGridFail(data, scope);
        }

        function setGriddingSuccess(data, scope) {
            setStep(2.2);
            $("#divGridStatus").text("Gridding was sucessful!");
            renderGrid(data, scope);
        }

        function doDetectColonyTask(scope, row, col, next) {
            scope.PlateColonyNextTaskInQueue = next;
            detectColony(scope, row, col);
        }

        function detectColony(scope, row, col) {
            scope.PlateCurrentColonyRow = row;
            scope.PlateCurrentColonyCol = col;
            SetColonyDetection(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken, row, col, setColonyDetectionSuccess, setColonyDetectionError);
        }

        function createDetectColonyTask(scope, row, col) {
            return function (next) {
                doDetectColonyTask(scope, row, col, next);
            }
        }

        function moveProgress(row, col, totalRow, totalCol) {
            var m = (row * totalCol) + col;
            var percent = (m * 100) / (totalCol * totalRow);
            $("#progressbar").progressbar({ value: percent });
            $("#divPro").text("col:"+col+" row:"+row+" m: "+m +" %:"+percent);
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
                setStep(3.1);

                var canvasBg = document.getElementById("colonyImageCanvas");
                var canvasFg = document.getElementById("colonyMarkingsCanvas");
                var canvasMarkings = document.getElementById("colonyPlotCanvas");

                createCanvasMarker(colony, canvasMarkings);
                createCanvasImage(colony, canvasBg);

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

            setStep(3.2);
            metaDataCanvasState.addShape(new Blob(canvasBg.width / 2, canvasBg.height / 2, 15, "rgba(255, 0, 0, .2)"));
        });

        function updateColonyData(blob, background, updateBlob, updatedBackground) {
            for (var row = 0; row < blob.length; row++) {
                for (var col = 0; col < blob[0].length; col++) {
                    var valueBlob = blob[row][col];
                    var valuebackground = background[row][col];
                    if (!valueBlob && !valuebackground)
                        continue;
                    blob[row][col] = updateBlob[row][col];
                    background[row][col] = updatedBackground[row][col];
                }
            }
            var newMetadata = {
                blob: blob,
                background: background
            }
            return newMetadata;
        }

        $("#btnSetUpdatedColonyMetadata").click(function () {
            var canvasMarkings = document.getElementById("colonyPlotCanvas");
            var newMetaData = getMarkerData("colonyMarkingsCanvas");
            var scope = getCurrentScope();

            var updatedMetaData = updateColonyData(scope.PlateCurrentColony.blob, scope.PlateCurrentColony.background, newMetaData.blob, newMetaData.background);
            scope.PlateCurrentColony.blob = updatedMetaData.blob;
            scope.PlateCurrentColony.background = updatedMetaData.background;

            createCanvasMarker(scope.PlateCurrentColony, canvasMarkings);
            setStep(3.1);
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
            SetColonyCompressionV2(scope, scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken, scope.PlateCurrentColony, scope.PlateCurrentColonyRow, scope.PlateCurrentColonyCol, setColonyCompressionSuccess, setColonyCompressionError);
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
            if (typeof reason == 'undefined')
                reason = '';
            var item = success + "- " + reason;
            scope.ColonyDetection.push(item);
            if (success)
                scope.ColonyDetected = scope.ColonyDetected + 1;
            else
                scope.ColonyNotDetected = scope.ColonyNotDetected + 1;
            $("#divCompress").text("Detected:" + scope.ColonyDetected + " fails:" + scope.ColonyNotDetected + " last error:" + data.reason);
        }

        function reportColonyCompressionProgress(data) {
            var success = data.success;
            var reason = data.reason;
            if (typeof reason == 'undefined')
                reason = '';
            var item = success + "- " + reason;
            scope.ColonyCompression.push(item);
            if (success)
                scope.ColonyCompressed = scope.ColonyCompressed + 1;
            else
                scope.ColonyNotCompressed = scope.ColonyNotCompressed + 1;
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
    });
