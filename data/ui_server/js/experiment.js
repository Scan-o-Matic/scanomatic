function validate_experiment() {

}

var current_fixture_id;
var fixture_selected = false;
var fixture_plates = [];

function update_fixture(options) {
    var fixture = $(options).val();

    $.get("/fixtures/" + fixture, function(data, status) {
        if (data.success) {
            fixture_plates = data.plates;
            set_pining_options_from_plates(data.plates);
            fixture_selected = true;
        } else {
            fixture_plates = [];
            console.log(data.reason);
            set_pining_options_from_plates(fixture_plates);
            fixture_selected = false;
        }

    });
}

function set_pining_options_from_plates(plates) {

}