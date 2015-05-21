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
var scale = 1;
var areas = [];
var creatingArea = false;

function relMouseCoords(event){
    var totalOffsetX = 0;
    var totalOffsetY = 0;
    var canvasX = 0;
    var canvasY = 0;
    var currentElement = this;

    do{
        totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
        totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
    }
    while(currentElement = currentElement.offsetParent)

    canvasX = event.pageX - totalOffsetX;
    canvasY = event.pageY - totalOffsetY;

    return {x:canvasX, y:canvasY}
}

function translateToImageCoords(coords) {
    var imageCoords = JSON.parse(JSON.stringify(coords));
    imageCoords.x /= scale;
    imageCoords.y /= scale;
    return imageCoords;
}

HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;

function set_canvas() {

    var selected_fixture_canvas_jq = $(selected_fixture_canvas_id);
    var selected_fixture_canvas = selected_fixture_canvas_jq[0];

    selected_fixture_canvas_jq.mousedown(function (event) {
        var canvasPos = selected_fixture_canvas.relMouseCoords(event);
        var imagePos = translateToImageCoords(canvasPos);
        creatingArea = pointInsideOther(imagePos);
        if (creatingArea < 0) {
            areas.push({
                x1: imagePos.x,
                x2: imagePos.x,
                y1: imagePos.y,
                y2: imagePos.y,
                grayscale: false,
                plate: -1
            });
            creatingArea = areas.length - 1;
        } else if (event.witch == 1) {
            areas[creatingArea].x2 = imagePos.x;
            areas[creatingArea].y2 = imagePos.y;
        } else {
            areas.splice(creatingArea, 1);
            creatingArea = null;
        }
        draw_fixture();
        setPlateIndices();
    });

    selected_fixture_canvas_jq.mousemove(function (event) {
        if (creatingArea && creatingArea >= 0 && creatingArea < areas.length) {
            var canvasPos = selected_fixture_canvas.relMouseCoords(event);
            var imagePos = translateToImageCoords(canvasPos);
            areas[creatingArea].x2 = imagePos.x;
            areas[creatingArea].y2 = imagePos.y;
            draw_fixture();
        }
    });

    selected_fixture_canvas_jq.mouseup( function(event) {
        var minUsableSize = 50;
        if (creatingArea && creatingArea >= 0 && creatingArea < areas.length) {
            if (getAreaSize(creatingArea) < minUsableSize)
                areas.splice(creatingArea, 1);
            else {
                var area = JSON.parse(JSON.stringify(areas[creatingArea]));
                area.x1 = Math.min(areas[creatingArea].x1, areas[creatingArea].x2);
                area.x2 = Math.max(areas[creatingArea].x1, areas[creatingArea].x2);
                area.y1 = Math.min(areas[creatingArea].y1, areas[creatingArea].y2);
                area.y2 = Math.max(areas[creatingArea].y1, areas[creatingArea].y2);
                areas[creatingArea] = area;
            }
            draw_fixture();

        }
        creatingArea = null;
     });

}

function getAreaSize(index) {
    if (index && index >= 0 && index < areas.length)
        return Math.abs(areas[index].x1 - areas[index].x2) * Math.abs(areas[index].y1 - areas[index].y2);
    return -1;
}

function setPlateIndices() {

}

function pointInsideOther(point) {
    for (var len = areas.length, i=0; i<len; i++) {
        if (areas[i].x1 < point.x  && point.x < areas[i].x2 &&
            areas[i].y1 < point.y && point.y < areas[i].y2)

            return i;
    }
    return -1;
}

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

    context.clearRect(0, 0, canvas.width, canvas.height);

    if (fixture_image) {
        scale = get_updated_scale(canvas, fixture_image);
        context.drawImage(fixture_image, 0, 0, fixture_image.width * scale, fixture_image.height * scale);
    }

    if (markers) {
        var radius = 30 * scale;
        var marker_scale = 4;
        for (var len = markers.length, i=0; i<len;i++)
            draw_marker(context, markers[i][0] * scale * marker_scale,
                        markers[i][1] * scale * marker_scale, radius, "blue", 5);
    }

    if (areas) {
        for (var i=0; i<areas.length; i++)
            draw_plate(context, areas[i]);
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

function get_updated_scale(canvas, obj) {
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

function draw_plate(context, plate) {
    context.beginPath();
    context.rect(plate.x1 * scale, plate.y1 * scale, (plate.x2 - plate.x1) * scale, (plate.y2 - plate.y1) * scale);
    context.fillStyle = "rgba(0, 255, 0, 0.3)";
    context.fill();
    context.strokeStyle = "green";
    context.lineWidth = 2;
    context.stroke();
}