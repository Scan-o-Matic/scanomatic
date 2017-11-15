import React from 'react';
import ReactDOM from 'react-dom';

import {
    GetFixtures,
    GetPinningFormats,
    InitiateCCC,
} from './api';

import CCCEditorContainer from './containers/CCCEditorContainer';



window.cccFunctions = {
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

    ini();

    function ini() {
        $("#divIniCCC").hide();
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
        dialogCCCIni.dialog("close");
        ReactDOM.render(
            <CCCEditorContainer
                cccId={id}
                pinFormat={pinFormat.split(',').map((i) => parseInt(i))}
                fixtureName={fixture}
                accessToken={token}
                onFinish={() => alert('Level completed!')}
            />,
            document.getElementById('react-root'),
        );
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
};


$(document).ready(executeCCC);

export default cccFunctions;
