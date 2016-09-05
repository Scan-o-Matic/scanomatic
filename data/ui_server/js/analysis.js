var gridplates = null;
var localFixture = false;

function toggleLocalFixture(caller) {
    localFixture = $(caller).prop("checked");
    InputEnabled($(current_fixture_id), !localFixture);
    set_fixture_plate_listing();
}

function set_fixture_plate_listing() {
    callback = function(data, status) {
        if (!data.success) {
            $("#fixture-error-message").html("<em>" + data.reason + "</em>").show();
            $("#manual-regridding-settings").hide();
         } else {
            $("#fixture-error-message").hide();
            gridplates = Map(data.plates, function (e) {return e.index;});
            if ($("#manual-regridding").prop("checked")) {
                $("#manual-regridding-settings").show();
            } else {
                $("#manual-regridding-settings").hide();
            }
            parent = $("#manual-regridding-plates");
            parent.empty();
            Map(gridplates, function(e) {append_regridding_ui(parent, e);});
         }
    };

    error_callback = function() {
        $("#fixture-error-message").html("<em>Fixture file missing</em>").show();
    }

    if (localFixture) {
        $.get("/api/data/fixture/local/" + path.substring(5, path.length), callback).fail(error_callback);
    } else {
        fixt = $(current_fixture_id).val();
        if (fixt) {
            $.get("/api/data/fixture/get/" + fixt, callback).fail(error_callback);
        } else {
            $("#fixture-error-message").hide();
        }
    }
}

function append_regridding_ui(parent, plate_index) {
    parent.append(
        "<div class='plate-regridding' id='plate-regridding-" + plate_index + "'>" +
            "<fieldset>" +
            "<img class='grid_icon' src='/images/grid_icon.png' onmouseenter='loadgridimage(" + plate_index + ");' onmouseexit='hidegridimage();'>" +
            "<legend>Plate " +  plate_index + "</legend>" +

            "<input type='radio' name='plate-regridding-radio-" + plate_index + "' value='Keep' checked='checked'>" +
            "<label id='plate-regridding-keep" + plate_index + "'>Keep previous</label><br>" +

            "<input type='radio' name='plate-regridding-radio-" + plate_index + "' value='Offset'>" +
            "<label id='plate-regridding-offset" + plate_index + "'>Offset</label>" +
            "<input type='number' class='plate-offset' id='plate-regridding-offset-d1-" + plate_index + "' value='0' name='Offset-d1'>" +
            "<input type='number' class='plate-offset' id='plate-regridding-offset-d2-" + plate_index + "' value='0' name='Offset-d2'><br>" +

            "<input type='radio' name='plate-regridding-radio-" + plate_index + "' value='New'>" +
            "<label id='plate-regridding-new" + plate_index + "'>New grid from scratch</label><br>" +
            "</fieldset>" +
        "</div>"
    );
}

function can_set_regridding() {
    return true;
    //    return gridplates != null && gridplates.length > 0;
}

function regridding_settings_data() {
    max = Math.max.apply(Math, gridplates);
    plates = [];
    for (i=1;i<=max; i++) {
        plates.push(get_regridding_setting(i));
    }
    return plates;
}

function get_regridding_setting(i) {

    e = $("#plate-regridding-" + i);
    if (e.length != 0) {
        switch (e.find("input[name=plate-regridding-radio-" + i + "]:checked").val()) {
            case "Keep":
                return [0, 0];
            case "Offset":
                return [
                    parseInt(e.find("#plate-regridding-offset-d1-" + i).val()),
                    parseInt(e.find("#plate-regridding-offset-d2-" + i).val()),
                ];

            case "New":
                return null;
            default:
                return null;
        };
    } else {
        return null;
    }
}

function set_regridding_source_directory(input) {

    get_path_suggestions(
        input,
        true,
        "",
        function(data, status) {

            //TODO: For some reason popup don't appear...

            regrid_chkbox = $("#manual-regridding");
            regrid_chkbox
                .prop("disabled", data.has_analysis ? false : true)
                .prop("checked", data.has_analysis ? true : false);

            toggleManualRegridding(regrid_chkbox);

        },
        path,
        true);
}

function toggleManualRegridding(chkbox) {
    is_active = $(chkbox).prop("checked");
    if (is_active && can_set_regridding()) {
        $("#manual-regridding-settings").show();
    } else {
        $("#manual-regridding-settings").hide();
    }
}

function set_analysis_directory(input, validate) {

    get_path_suggestions(
        input,
        true,
        "",
        function(data, status) {
            if (validate) {
                InputEnabled($("#submit-button2"), data.valid_parent && data.exists);
            }

            if (localFixture) {
                set_fixture_plate_listing();
            }

    });
}

function set_file_path(input, suffix) {

    get_path_suggestions(
        input,
        false,
        suffix,
        function(data, status) {
    });
}

function Analyse(button) {

    InputEnabled($(button), false);

    data = {
            compilation: $("#compilation").val(),
            compile_instructions: $("#compile-instructions").val(),
            output_directory: $("#analysis-directory").val(),
            chain: $("#chain-analysis-request").is(':checked') ? 0 : 1,
            one_time_positioning: $("#one_time_positioning").is(':checked') ? 0 : 1,
    };

    if ($("#manual-regridding").prop("checked")) {
        data['killl']
    }


    $.ajax({
        url: '?action=analysis',
        data: data,
        method: 'POST',
        success: function(data) {
            if (data.success) {
                Dialogue("Analysis", "Analysis Enqueued", "", "/status");
            } else {
                Dialogue("Analysis", "Analysis Refused", data.reason ? data.reason : "Unknown reason", false, button);
            }
        },
        error: function(data) {
            Dialogue("Analysis", "Error", "An error occurred processing request", false, button);
        }

    });
}

function Extract(button) {
    InputEnabled($(button), false)

    $.ajax({
        url: '?action=extract',
        data: {
            analysis_directory: $("#extract").val()
               },
        method: 'POST',
        success: function(data) {
            if (data.success) {
                Dialogue("Feature Extraction", "Extraction Enqueued", "", "/status");
            } else {
                Dialogue("Feature Extraction", "Extraction refused", data.reason ? data.reason : "Unknown reason", false, button);
            }
        },
        error: function(data) {
            Dialogue("Feature Extraction", "Error", "An error occurred processing request", false, button);
        }

    });
}