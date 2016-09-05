
var localFixture = false;
var path = '';
var project_path_valid = false;
var image_list_div = null;
var gridplates = null;

function set_project_directory(input) {

    get_path_suggestions(
        input,
        true,
        "",
        function(data, status) {
            path = $(input).val();
            project_path_valid = data.valid_parent && data.exists;
            $("#manual-regridding-source-folder").prop("disabled", !project_path_valid);
            if (project_path_valid) {
                setImageSuggestions(path);
                InputEnabled(image_list_div.find("#manual-selection"), true);
            } else {
                toggleManualSelection(false);
                InputEnabled(image_list_div.find("#manual-selection"), false);
            }

            if (localFixture) {
                set_fixture_plate_listing();
            }

            InputEnabled($("#submit-button"), project_path_valid);
    });
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

function setImageSuggestions(path) {

    //Only do stuff if path changed
    if (image_list_div.find("#hidden-path").val() != path)
    {
        image_list_div.find("#hidden-path").val(path);

        image_list_div.find("#manual-selection").prop("checked", false);

        options = image_list_div.find("#options");
        options.empty();

        $.get("/api/compile/image_list/" + path, function(data, status)
        {
            for (var i=0; i<data.images.length; i++)
            {
                row_class = i % 2 == 0 ? 'list-entry-even' : 'list-entry-odd';
                image_data = data.images[i];
                options.append(
                    "<div class='" + row_class + "'>" + String('00' + image_data.index).slice(-3) + ": " +
                    "<input type='checkbox' id='image-data-" + image_data.index + "' checked='checked' value='" + image_data.file + "'>" +
                    "<label class='image-list-label' for='image-data-" + image_data.index + "'>" + image_data.file + "</label></div>");
            }

        });

    }
    else
    {
        toggleManualSelectionBtn(image_list_div.find("#manual-selection"));
    }
}

function toggleManualSelectionBtn(button) {
    toggleManualSelection($(button).prop("checked"));
}

function toggleManualSelection(is_manual) {
    if (is_manual)
    {
        image_list_div.find("#options").show();
        image_list_div.find("#list-buttons").show();
    }
    else
    {
        image_list_div.find("#options").hide();
        image_list_div.find("#list-buttons").hide();
    }
}

function setOnAllImages(included) {
    image_list_div.find("#options").children().each(function () {
        $(this).find(":input").prop("checked", included);
    });
}


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
    return gridplates != null && gridplates.length > 0;
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

function Compile(button) {

    InputEnabled($(button), false);

    images = null;
    if (image_list_div.find("#manual-selection").prop("checked")) {
        images = [];
        image_list_div.find("#options").children().each(function() {
            imp = $(this).find(":input");
            if (imp.prop("checked") == true) {
                images.push(imp.val());
            }
        });
    }

    data = {local: localFixture ? 1 : 0, 'fixture': $(current_fixture_id).val(),
               path: path,
               chain: $("#chain-analysis-request").is(':checked') ? 0 : 1,
               images: images
            };

    if ($("#manual-regridding").prop("checked")) {
        data['']
    }

    $.ajax({
        url: "?run=1",
        method: "POST",
        data: data,
        success: function (data) {
            if (data.success) {
                Dialogue("Compile", "Compilation enqueued", "", '/status');
            } else {
                Dialogue("Compile", "Compilation Refused", data.reason ? data.reason : "Unknown reason", false, button);
            }

        },
        error: function(data) {
            Dialogue("Compile", "Error", "An error occurred processing the request.", false, button);
        }});
}