function validate_experiment() {

}

var current_fixture_id;
var fixture_selected = false;
var fixture_plates = [];
var description_cache = {};
var duration;
var interval;

function update_fixture(options) {
    var fixture = $(options).val();

    $.get("/fixtures/" + fixture, function(data, status) {
        if (data.success) {
            fixture_selected = true;
        } else {
            data.plates = [];
            console.log(data.reason);
            fixture_selected = false;
        }
        fixture_plates = data.plates.slice();
        set_pining_options_from_plates(data.plates);
        set_visible_plate_descriptions();
    });
}

function set_visible_plate_descriptions() {
    var active_indices = $.map(fixture_plates.filter(function (value) { return value.active === true;}), function(entry, index) {return  entry.index;});

    var descriptions = $("#plate-descriptions");
    if (fixture_plates.length == 0) {
        descriptions.html("<em>Please, select a fixture and plate pinnings first.</em>");
        return;
    } else
        descriptions.html("");

    for (var i=0; i<active_indices.length; i++) {
        var index = active_indices[i];
        descriptions.append(get_description(index, description_cache[index] !== undefined ? description_cache[index] : ""));
    }
}

function cache_description(target) {
    var index = get_fixture_plate_from_obj($(target).parent());
    description_cache[index] = $(target).val();
}

function set_active_plate(plate) {
    var index = get_fixture_plate_index_ordinal(get_fixture_plate_from_obj($(plate).parent()));
    fixture_plates[index].active = $(plate).val() != "";
    set_visible_plate_descriptions();
}

function get_fixture_plate_from_obj(obj) {
    return parseInt(obj.find("input[type='hidden']").val());
}

function get_fixture_plate_index_ordinal(index) {
    for (var i=0; i<fixture_plates.length;i++) {
        if (fixture_plates[i].index == index)
            return i;
    }
    return -1;
}

function set_pining_options_from_plates(plates) {
    var active_indices = $.map(plates, function(entry, index) {return  entry.index;});
    var pinnings = $("#pinnings");
    var pinning_list = $(".pinning")

    if (plates.length == 0) {
        pinnings.html("<em>Please select a fixture first.</em>");
        return;
    } else if (pinning_list.length == 0) {
        pinnings.html("");
    }

    pinning_list.each(function() {
        var options = $(this).find("select");
        var index = parseInt($(this).select("input[type='hidden']").val());
        if (!(index in active_indices))
            $(this).remove();
        else {
            while (plates.length > 0 && plates[0].index <= index) {
                var plate = plates.shift();
                if (plate.index != index) {
                    $(this).before(get_plate_selector(plate));
                }
            }
        }

    });
    for (var i=0; i<plates.length; i++) {
        pinnings.append(get_plate_selector(plates[i]));
        fixture_plates[get_fixture_plate_index_ordinal(plates[i].index)].active = true;
    }
}

function format_time(input) {
    var s = $(input).val();

    try {
        var days = parseInt(s.match(/(\d+) ?(days|d)/i)[1]);
    } catch(err) {
        var days = 0;
    }

    try {
        var minutes = parseInt(s.match(/(\d+) ?(min|minutes?|m)/i)[1]);
    } catch(err) {
        var minutes = 0;
    }

    try {
        var hours = parseInt(s.match(/(\d+) ?(hours?|h)/i)[1]);
    } catch(arr) {

        if (minutes == 0 && days == 0) {

            var hours = parseFloat(s);
            if (isNaN(hours))
                hours = 0;


            minutes = Math.round((hours - Math.floor(hours)) * 60);
            hours = Math.floor(hours);
        } else
            var hours = 0;
    }
    if (minutes > 59) {
        hours += Math.floor(minutes / 60);
        minutes = minutes % 60;
    }
    if (hours > 23) {
        days += Math.floor(hours / 24);
        hours %= 24;
    }

    if (days == 0 && hours == 0 && minutes == 0)
        days = 3;

    duration = [days, hours, minutes];
    $(input).val(days + " days, " + hours + " hours, " + minutes + " minutes");
}

function format_minutes(input) {
    interval = parseFloat($(input).val());
    if (isNaN(interval) || interval == 0)
        interval = 20.0;

    $(input).val(interval + " minutes");
}

function get_plate_selector(plate) {
    return "<div class='pinning'><input type='hidden' value='" + plate.index + "'>" +
                    "<label for='pinning-plate-" + plate.index + "'>Plate " + plate.index + "</label>" +
                    "<select id='pinning-plate-" + plate.index + "' class='pinning-selector' onchange='set_active_plate(this); validate_experiment();'>" +
                        "<option value=''>Not used</option>" +
                        "<option value='96'>8 x 12 (96)</option>" +
                        "<option value='384'>16 x 24 (384)</option>" +
                        "<option value='1536' selected>32 x 48 (1536)</option>" +
                        "<option value='6144'>64 x 96 (6144)</option>" +

                    "</select></div>"
}

function get_description(index, description) {
    return "<div class='plate-description'>" +
        "<input type='hidden' value='" + index + "'>" +
        "<label for='plate-description-" + index + "'>Plate " + index + "</label>" +
        "<input class='long' id='plate-description-" + index + "' value='" + description + "' onchange='cache_description(this);'></div>";
}