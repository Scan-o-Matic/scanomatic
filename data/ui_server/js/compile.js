
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
    if (is_active) {
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

    $.ajax({
        url: "?run=1",
        method: "POST",
        data: {local: localFixture ? 1 : 0, 'fixture': $(current_fixture_id).val(),
               path: path,
               chain: $("#chain-analysis-request").is(':checked') ? 0 : 1,
               images: images
               },
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