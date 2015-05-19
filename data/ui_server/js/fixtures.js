var current_fixture_id;
var new_fixture_data_id;
var new_fixture_detect_id;
var new_fixture_image_id;

function get_fixture_as_name(fixture) {
    return fixture.replace("_", " ")
        .replace(/^[a-z]/g,
            function ($1) { return $1.toUpperCase();});
}

function unselect(target) {
    target.val("");
}

function get_fixtures() {
    var options = $(current_fixture_id);
    options.empty();
    $.get("/fixtures?names=1", function(data, status) {
        $.each(data.split(","), function() {
            options.append($("<option />").val(this).text(get_fixture_as_name(this)));
        })
        unselect(options);
    })

    $(new_fixture_data_id).hide();
}

function add_fixture() {
    var options = $(current_fixture_id)
    unselect(options);
    set_fixture_image();
    $(new_fixture_data_id).show()
}

function get_fixture() {
    var options = $(current_fixture_id);
    $(new_fixture_data_id).hide();
}

function set_fixture_image() {
    if ($(new_fixture_image_id).val() == null)
        $(new_fixture_detect_id).attr("disabled", "disabled");
    else
        $(new_fixture_detect_id).removeAttr("disabled");
}