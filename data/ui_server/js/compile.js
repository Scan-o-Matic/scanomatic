
var localFixture = false;
var path = '';
var project_path_valid = false;

function set_project_directory(input) {

    get_path_suggestions(
        input,
        true,
        "",
        function(data, status) {
            path = $(input).val();
            project_path_valid = data.valid_parent && data.exists;
            InputEnabled($("#submit-button"), project_path_valid);
    });
}

function toggleLocalFixture(caller) {
    localFixture = $(caller).prop("checked");
    InputEnabled($(current_fixture_id), !localFixture);
}


function Compile(button) {

    InputEnabled($(button), false);

    $.ajax({
        url: "?run=1",
        method: "POST",
        data: {local: localFixture ? 1 : 0, 'fixture': $(current_fixture_id).val(),
               path: path,
               chain: $("#chain-analysis-request").is(':checked') ? 0 : 1,
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