import React from 'react';
import ReactDOM from 'react-dom';

import {
    GetFixtures,
    GetFixturePlates,
    GetPinningFormats,
    InitiateCCC,
    SetCccImageData,
    SetCccImageSlice,
    SetGrayScaleImageAnalysis,
    SetGrayScaleTransform,
    GetImageId,
    GetMarkers,
} from './api';
import { createScope, getCurrentScope, setCurrentScope } from './scope';
import PlateEditorContainer from './containers/PlateEditorContainer';



window.cccFunctions = {
    setStep: (step) => {
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
        default:
        }
    },
    initiateNewCcc: (species, reference, allFields) => {
        let valid = true;
        const validNameRegexp = /^[a-z]([0-9a-z_\s])+$/i;
        const invalidNameMsg = "This field may consist of a-z, 0-9, underscores and spaces, and must begin with a letter.";

        allFields.removeClass("ui-state-error");

        valid = valid && cccFunctions.checkLength(
            species, 3, 20, "species");
        valid = valid && cccFunctions.checkLength(
            reference, 3, 20, "reference");

        valid = valid && cccFunctions.checkName(
            species, validNameRegexp, invalidNameMsg);
        valid = valid && cccFunctions.checkName(
            reference, validNameRegexp, invalidNameMsg);

        if (valid) {
            InitiateCCC(
                species.val(),
                reference.val(),
                cccFunctions.initiateCccSuccess,
                cccFunctions.initiateCccError);
        }
        return valid;
    },
    initiateCccError: (data) => {
        cccFunctions.updateTips(data.responseJSON.reason);
    },
    checkLength: (obj, min, max, field) => {
        if (obj.val().length > max || obj.val().length < min) {
            obj.addClass("ui-state-error");
            cccFunctions.updateTips(
                `Length of ${field} must be between ${min} and ${max}.`);
            return false;
        } else {
            return true;
        }
    },
    checkName: (obj, regexp, message) => {
        if (!(regexp.test(obj.val()))) {
            obj.addClass("ui-state-error");
            cccFunctions.updateTips(message);
            return false;
        } else {
            return true;
        }
    }
};

window.executeCCC = function() {
    var selFixtureName = "selFixtures";
    var selPinFormatsName = "selPinFormats";
    var dialogCCCIni;
    var form;
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
            "Initiate new CCC": () => cccFunctions.initiateNewCcc(
                species, reference, allFields),
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
                alert("ERROR: There was a problem with the API while fetching pinnnig formats!");
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
            500
        );
    }

    cccFunctions.updateTips = updateTips;

    function initiateCccSuccess(data) {
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
        cccFunctions.setStep(1.2);
        GetMarkers(scope.FixtureName, scope.File)
            .then(data => getMarkersSuccess(data, scope), getMarkersError);
    }

    function getMarkersError(reason) {
        alert("Markers error:" + reason);
    }

    function getMarkersSuccess(data, scope) {
        cccFunctions.setStep(1.3);
        scope.Markers = processMarkers(data.markers);
        var file = scope.File;
        scope.File = null;
        GetImageId(scope.cccId, file, scope.AccessToken)
            .then(data => getImageIdSuccess(data, scope), getImageIdError);
    }

    function getImageIdError(reason) {
        alert("Fatal error uploading the image: \n " + reason);
    }

    function getImageIdSuccess(data, scope) {
        cccFunctions.setStep(1.4);
        scope.CurrentImageId = data.image_identifier;
        var markers = scope.Markers;
        var toSetData = [];
        toSetData.push({ key: "marker_x", value: markers[0] });
        toSetData.push({ key: "marker_y", value: markers[1] });
        SetCccImageData(scope.cccId, scope.CurrentImageId, scope.AccessToken, toSetData, scope.FixtureName)
            .then(data => setCccImageDataSuccess(data, scope), setCccImageDataError);
    }

    function setCccImageDataError(reason) {
        alert("Error while setting up the images! " + reason);
    }

    function setCccImageDataSuccess(data, scope) {
        cccFunctions.setStep(1.5);
        SetCccImageSlice(scope.cccId, scope.CurrentImageId, scope.AccessToken)
            .then(data => setCccImageSliceSuccess(data, scope), setCccImageSliceError);
    }

    function setCccImageSliceError(reason) {
        alert("Error while setting up the images slice!" + reason);
    }

    function setCccImageSliceSuccess(data, scope) {
        cccFunctions.setStep(1.6);
        SetGrayScaleImageAnalysis(scope.cccId, scope.CurrentImageId, scope.AccessToken)
            .then(data => setGrayScaleImageAnalysisSuccess(data, scope), setGrayScaleImageAnalysisError);
    }

    function setGrayScaleImageAnalysisError(reason) {
        alert("Error while starting grayscale analysis! " + reason);
    }

    function setGrayScaleImageAnalysisSuccess(data, scope) {
        cccFunctions.setStep(1.7);
        //store target_values and source_values to QC graph ???
        GetFixturePlates(scope.FixtureName)
            .then(data => createFixturePlateSelection(data, scope))
            .catch(alert);
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
            cccFunctions.setStep(2);
            scope.Plate = plate;
            scope.PlateNextTaskInQueue = next;
            SetGrayScaleTransform(scope.cccId, scope.CurrentImageId, scope.Plate, scope.AccessToken)
                .then(data => setGrayScaleTransformSuccess(data, scope), setGrayScaleTransformError);
        };
    }

    cccFunctions.createSetGrayScaleTransformTask = createSetGrayScaleTransformTask;

    function setGrayScaleTransformError(reason) {
        alert("set grayscale transform error:" + reason);
    }

    cccFunctions.setGrayScaleTransformError = setGrayScaleTransformError;

    function setGrayScaleTransformSuccess(data, scope) {
        ReactDOM.render(
            <PlateEditorContainer
                cccId={scope.cccId}
                imageId={scope.CurrentImageId}
                plateId={scope.Plate}
                pinFormat={scope.PinFormat.map((i) => parseInt(i))}
                accessToken={scope.AccessToken}
                onFinish={() => alert('Level completed!')}
            />,
            document.getElementById('react-root'),
        );
    }

    cccFunctions.setGrayScaleTransformSuccess = setGrayScaleTransformSuccess;
};


$(document).ready(executeCCC);

export default cccFunctions;
