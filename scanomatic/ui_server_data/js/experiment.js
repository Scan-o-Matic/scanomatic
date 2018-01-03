var current_fixture_id;
var fixture_selected = false;
var fixture_plates = [];
var description_cache = {};
var duration = [0,0,0];
var interval = 20.0;
var project_path;
var project_path_valid = false;
var project_path_invalid_reason = '';
var poetry = "If with science, you let your life deteriorate,\nat least do it right.\nDo it with plight\nand fill out this field before it's too late";
var auxillary_info = {};
var auxillary_info_time_data = {};
var number_of_scans = -1;

function validate_experiment() {
    set_validation_status("#project-path", project_path_valid, project_path_valid ? 'Everything is cool' : project_path_invalid_reason);
    var scanner = $("#current-scanner");
    var scanner_ok = scanner.val() != '' && scanner.val() != undefined && scanner.val() != null;
    set_validation_status(scanner, scanner_ok, scanner_ok ? 'The scanner is yours' :
        'It would seem likely that you need a scanner... to scan');
    var active_plates = fixture_plates.filter(function(e, i) { return e.active;}).length > 0;
    set_validation_status("#current-fixture", fixture_selected && active_plates,
        !fixture_selected ? "Select fixture" : (active_plates ? "No plates active would mean doing very little" : "All good!"));
    var time_makes_sense = get_duration_as_minutes() >= 0;
    set_validation_status("#project-duration", time_makes_sense, time_makes_sense ? "That's a splendid project duration" : "Nope, you can't reverse time");

    InputEnabled($("#submit-button"), project_path_valid && scanner_ok && fixture_selected && active_plates && time_makes_sense);
}

function set_validation_status(dom_object, is_valid, tooltip) {
    var input = $(dom_object);
    var sib = input.next();
    if (sib.attr("class") !== 'icon-large')
        sib = $(get_validation_image()).insertAfter(input);

    if (is_valid)
        sib.attr('src', '/images/yeastOK.png').attr('alt',tooltip).attr('title', tooltip);
    else
        sib.attr('src', '/images/yeastNOK.png').attr('alt', tooltip).attr('title', tooltip);
}

function set_poetry(input) {

    $(input).html(poetry).css('font-style', 'italic');

    $(input).focus(function() {

        if ($(this).html() === poetry)
            $(this).html('').css('font-style', 'normal');
    });

    $(input).blur(function() {
        if ($(this).html() === '') {
            $(this).html(poetry).css('font-style', 'italic');
        }
    });
}

function set_experiment_root(input) {

    get_path_suggestions(
        input,
        true,
        "",
        null,
        function(data, status) {
            project_path = $(input).val();
            project_path_valid = data.valid_parent && !data.exists;
            project_path_invalid_reason = data.reason;

            set_validation_status("#project-path", project_path_valid, project_path_valid ? 'Everything is cool' : project_path_invalid_reason);
            if (data.valid_parent && !data.exists)
                $("#experiment-title").html("Start Experiment '" + data.prefix + "'");
            else
                $("#experiment-title").html("Start Experiment");
    });
}

function update_fixture(options) {
    var fixture = $(options).val();

    $.get("/api/data/fixture/get/" + fixture, function(data, status) {
        if (data.success) {
            fixture_selected = true;
        } else {
            data.plates = [];
            fixture_selected = false;
        }
        fixture_plates = data.plates.slice();
        set_pining_options_from_plates(data.plates);
        set_visible_plate_descriptions();
        validate_experiment();
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

function get_descriptions() {
        var maxIndex = Math.max.apply(null,
            Object.keys(description_cache).map(function (item) { return parseInt(item, 10);}));

        var ret = [];

        for (var i=0; i<maxIndex; i++)
            ret[i] = description_cache[i + 1] !== undefined && fixture_plates[i].active ? description_cache[i + 1] : null;

        return ret;
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
    if (isNaN(interval) || interval <= 0)
        interval = 20.0;
    else if (interval < 7)
        interval = 7.0;

    $(input).val(interval + " minutes");
}

function get_pinnings() {
    var ret = [];
    $(".pinning").each(
        function(i, e) {
            var idx = $(e).find("input[type=hidden]").first().val() - 1;
            if (fixture_plates[idx].active)
                ret[idx] = $(e)
                    .find(".pinning-selector").first().val()
                    .match(/\d+/g).map(function (elem) {return parseInt(elem, 10);});
            else
                ret[idx] = null;
        });

    return ret;
}

function update_scans(paragraph) {
    number_of_scans = Math.floor(get_duration_as_minutes() / interval) + 1;
    $(paragraph).html(number_of_scans);
}

function get_duration_as_minutes() {
    return (duration[0] * 24 + duration[1]) * 60 + duration[2];
}

function get_plate_selector(plate) {
    return "<div class='pinning'><input type='hidden' value='" + plate.index + "'>" +
                    "<label for='pinning-plate-" + plate.index + "'>Plate " + plate.index + "</label>" +
                    "<select id='pinning-plate-" + plate.index + "' class='pinning-selector' onchange='set_active_plate(this); validate_experiment();'>" +
                        "<option value=''>Not used</option>" +
                        "<option value='(8, 12)'>8 x 12 (96)</option>" +
                        "<option value='(16, 24)'>16 x 24 (384)</option>" +
                        "<option value='(32, 48)' selected>32 x 48 (1536)</option>" +
                        "<option value='(64, 96)'>64 x 96 (6144)</option>" +

                    "</select></div>"
}

function get_description(index, description) {
    return "<div class='plate-description'>" +
        "<input type='hidden' value='" + index + "'>" +
        "<label for='plate-description-" + index + "'>Plate " + index + "</label>" +
        "<input class='long' id='plate-description-" + index + "' value='" + description +
        "' placeholder='Enter medium and, if relevant, what experiment this plate is.' onchange='cache_description(this);'></div>";
}

function get_validation_image() {
    return "<img src='' class='icon-large'>";
}

function setAux(input, key) {
    auxillary_info[key] = $(input).val();
}

function setAuxTime(input, key, factor) {

    if (auxillary_info_time_data[key] === undefined)
        auxillary_info_time_data[key] = {}

    auxillary_info_time_data[key][factor] = parseFloat($(input).val());

    var total = 0;
    for (var k in auxillary_info_time_data[key])
        total += parseFloat(k) * auxillary_info_time_data[key][k];

    auxillary_info[key] = total;
}


function StartExperiment(button) {
    InputEnabled($(button), false);
    const data = {
        number_of_scans: number_of_scans,
        time_between_scans: interval,
        project_path: project_path,
        description: $('#project-description').val(),
        email: $('#project-email').val(),
        pinning_formats: get_pinnings(),
        fixture: $('#current-fixture').val(),
        scanner: parseInt($('#current-scanner').val()),
        plate_descriptions: get_descriptions(),
        cell_count_calibration_id: $('#ccc-selection').val(),
        auxillary_info: auxillary_info
    }

    API.postJSON('/api/project/experiment', data)
        .then((response) => {
            Dialogue(
                'Experiment',
                'Experiment ' + response.name + ' enqueued, now go ' +
                'get a cup of coffee and pat yourself on the back. Your work here is done.',
                '', '/status');
        })
        .catch((reason) => {
            if (reason) {
                Dialogue('Analysis', 'Analysis Refused', reason, false, button);
            } else {
                Dialogue('Analysis', 'Unexpected error', 'An error occurred processing request', false, button);
            }
        });
}
