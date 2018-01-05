var localFixture = false;
var path = '';
var project_path_valid = false;
var image_list_div = null;

function set_fixture_status() {
    callback = function(data, status) {
        if (!data.success) {
            $("#fixture-error-message").html("<em>" + data.reason + "</em>").show();
         } else {
            $("#fixture-error-message").hide();
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

function set_project_directory(input) {

    get_path_suggestions(
        input,
        true,
        "",
        null,
        function(data, status) {
            path = $(input).val();
            project_path_valid = data.valid_parent && data.exists;

            if (project_path_valid) {

                setImageSuggestions(path);
                $("#project-directory-info").html("Scan images in folder: " + GetIncludedImageList(true).length);
                InputEnabled(image_list_div.find("#manual-selection"), true);
            } else {
                toggleManualSelection(false);
                $("#project-directory-info").html("<em>The project directory is the directory that contains the images that were scanned.</em>");
                InputEnabled(image_list_div.find("#manual-selection"), false);
            }

            if (localFixture) {
                set_fixture_status();
            }
            InputEnabled($("#submit-button"), project_path_valid);
    });
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

            no_img_err = "<em>Not a project folder</em>";
            if (data.images.length == 0) {
                $("#fixture-error-message").html(no_img_err).show();
            } else if ($("#fixture-error-message").html() == no_img_err) {
                $("#fixture-error-message").html("").hide();
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


function compileToggleLocalFixture(caller) {
    localFixture = $(caller).prop("checked");
    set_fixture_status();
    InputEnabled($(current_fixture_id), !localFixture);
}

function GetIncludedImageList(force_list) {
    images = null;
    if (force_list || image_list_div.find("#manual-selection").prop("checked")) {
        images = [];
        image_list_div.find("#options").children().each(function() {
            imp = $(this).find(":input");
            if (imp.prop("checked") == true) {
                images.push(imp.val());
            }
        });
    }
    return images;
}

function Compile(button) {

    InputEnabled($(button), false);

    const data = {
        local: localFixture ? 1 : 0,
        fixture: localFixture ? '' : $(current_fixture_id).val(),
        path: path,
        chain: $('#chain-analysis-request').is(':checked') ? 0 : 1,
        images: GetIncludedImageList()
    };

    API.postJSON('/api/project/compile', data)
        .then(() => Dialogue('Compile', 'Compilation enqueued', '', '/status'))
        .catch((reason) => {
            if (reason) {
                Dialogue('Compile', 'Compilation Refused', reason, false, button);
            } else {
                Dialogue('Compile', 'Unexpected error', 'An error occurred processing the request.', false, button);
            }
        });
}
