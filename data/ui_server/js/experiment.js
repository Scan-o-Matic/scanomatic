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
    var active_indices = $.map(plates, function(entry, index) {return  entry.index;});
    console.log(active_indices);
    var pinnings = $("#pinnings");
    var pinning_list = $(".pinning")

    if (plates.length == 0) {
        pinnings.html("<em>Please select a fixture first.</em>");
        return;
    } else if (pinning_list.length == 0)
        pinnings.html("");

    pinning_list.each(function() {
        var options = $(this).find("select");
        var index = parseInt($(this).select("input[type='hidden']").val());
        if (!(index in active_indices))
            $(this).remove();
        else {
            while (plates.length() > 0 && plates[0].index <= index) {
                var plate = plates.shift();
                if (plate.index != index) {
                    $(this).before(get_plate_selector(plate));
                }
            }
        }

    });
    for (var i=0; i<plates.length; i++)
        pinnings.append(get_plate_selector(plates[i]));
}

function get_plate_selector(plate) {
    return "<div class='pinning'><input type='hidden' value='" + plate.index + "'>" +
                    "<label for='pinning-plate-" + plate.index + "'>Plate " + plate.index + "</label>" +
                    "<select id='pinning-plate-" + plate.index + "' class='pinning-selector' onchange='validate_experiment();'>" +
                        "<option value=''>Not used</option>" +
                        "<option value='96'>8 x 12 (96)</option>" +
                        "<option value='384'>16 x 24 (384)</option>" +
                        "<option value='1536' selected>32 x 48 (1536)</option>" +
                        "<option value='6144'>64 x 96 (6144)</option>" +

                    "</select></div>"
}