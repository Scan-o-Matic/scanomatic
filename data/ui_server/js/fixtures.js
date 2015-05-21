var current_fixture_id;
var new_fixture_data_id;
var new_fixture_detect_id;
var new_fixture_image_id;
var new_fixture_markers_id;
var selected_fixture_div_id;
var fixture_name_id;
var selected_fixture_canvas_id;
var new_fixture_name;

var context_warning = "";
var fixture_image = null;
var markers = null;

function get_fixture_as_name(fixture) {
    return fixture.replace(/_/g, " ")
        .replace(/^[a-z]/g,
            function ($1) { return $1.toUpperCase();});
}

function get_fixture_from_name(fixture) {
    return fixture.replace(/ /g, "_")
        .replace(/A-Z/g, function ($1) { return $1.toLowerCase();})
        .replace(/[^a-z1-9_]/g,"");
}

function position_string_to_array(pos_str) {
    return JSON.parse(pos_str.replace(/\(|\)/g, function ($1) { return $1 == "(" ? "[" : "]";}));
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
    $(selected_fixture_div_id).hide();
}

function add_fixture() {
    var options = $(current_fixture_id);
    unselect(options);
    unselect($(new_fixture_image_id));
    set_fixture_image();
    $(new_fixture_detect_id).val("Detect");
    $(new_fixture_data_id).show();
    $(selected_fixture_div_id).hide();
}

function get_fixture() {
    var options = $(current_fixture_id);
    $(new_fixture_data_id).hide();
    load_fixture(options.val());
}

function set_fixture_image() {
    if ($(new_fixture_image_id).val() == "") {
        $(new_fixture_detect_id).attr("disabled", true);
    } else {
        $(new_fixture_detect_id).removeAttr("disabled");
    }
}

function detect_markers() {

    var formData = new FormData();
    formData.append("markers", $(new_fixture_markers_id).val());
    formData.append("image", $(new_fixture_image_id)[0].files[0]);
    formData.append("name", $(new_fixture_name).val());
    $(new_fixture_detect_id).attr("disabled", true);
    $(new_fixture_detect_id).val("...");
    $.ajax({
        url: '?detect=1',
        type: 'POST',
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function (data) {
            if (data.image && data.markers) {
                context_warning = ""
                $(new_fixture_data_id).hide();
            } else {
                context_warning = "Name or image refused";
            }
             load_fixture_image(data.image);
             set_fixture_markers(data.markers);
        },
        error: function (data) {
            context_warning = "Marker detection failed";
            markers = null;
            draw_fixture();
        }});

    var new_image = $(new_fixture_image_id);
    load_fixture($(new_fixture_name).val());
}

function endsWith(str, suffix) {
    return str.indexOf(suffix, str.length - suffix.length) !== -1;
}

function update_fixture_name() {
    $(fixture_name_id).text(get_fixture_as_name($(new_fixture_name).val()));
}

function load_fixture(name, img_data) {
    $(fixture_name_id).text(get_fixture_as_name(name));
    $(selected_fixture_div_id).show();
    draw_fixture();
}

function load_fixture_image(image_name) {

    if (image_name) {

        var img = new Image;
        img.onload = function() {
            fixture_image = img;
            draw_fixture();
        }
        img.src = "?image=" + image_name;

    } else {
        fixture_image = null;
    }
}


function set_fixture_markers(data) {
    console.log(data);
    markers = position_string_to_array(data);
    if (markers.length ==0) {
        markers = null;
        $(new_fixture_data_id).show();
    }
    draw_fixture();
}

function draw_fixture() {

    var canvas =  $(selected_fixture_canvas_id)[0];
    var context = canvas.getContext('2d');
    var scale = 1.0;

    context.clearRect(0, 0, canvas.width, canvas.height);

    if (fixture_image) {
        scale = get_updated_scale(scale, canvas, fixture_image);
        context.drawImage(fixture_image, 0, 0, fixture_image.width * scale, fixture_image.height * scale);
    }

    if (markers) {
        var radius = 30 * scale;
        for (var len = markers.length, i=0; i<len;i++)
            draw_marker(context, markers[i][0] * scale, markers[i][1] * scale, radius, "blue", 5);
    }

    if (context_warning) {
        var x = canvas.width / 2;
        var y = canvas.height / 2;
        context.font = '20pt Calibri';
        context.textAlign = 'center';
        context.fillStyle = 'red';
        context.fillText(context_warning, x, y);
    }
}

function get_updated_scale(scale, canvas, obj) {
    var x_scale = canvas.width / obj.width;
    var y_scale = canvas.height / obj.height;
    return Math.min(scale, x_scale, y_scale);
}

function draw_marker(context, centerX, centerY, radius, color, lineWidth) {
    context.beginPath();
    context.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
    context.lineWidth = lineWidth;
    context.strokeStyle = color;
    context.stroke();
}