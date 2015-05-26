var current_fixture_id;
var new_fixture_data_id;
var new_fixture_detect_id;
var new_fixture_image_id;
var new_fixture_markers_id;
var selected_fixture_div_id;
var fixture_name_id;
var selected_fixture_canvas_id;
var new_fixture_name;
var grayscale_id;

var context_warning = "";
var fixture_image = null;
var fixture_name = null;
var markers = null;
var scale = 1;
var areas = [];
var creatingArea = false;
var selected_fixture_canvas_jq;
var selected_fixture_canvas;

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

    selected_fixture_canvas_jq = $(selected_fixture_canvas_id);
    selected_fixture_canvas_jq.attr("tabindex", "0");
    selected_fixture_canvas = selected_fixture_canvas_jq[0];

    selected_fixture_canvas_jq.mousedown(function (event) {

        if (context_warning) {
            context_warning = null;
            return;
        }

        var canvasPos = selected_fixture_canvas.relMouseCoords(event);
        var imagePos = translateToImageCoords(canvasPos);
        creatingArea = null;
        var nextArea = getAreaByPoint(imagePos);
        if (event.button == 0) {
            if (nextArea < 0) {
                areas.push({
                    x1: imagePos.x,
                    x2: imagePos.x,
                    y1: imagePos.y,
                    y2: imagePos.y,
                    grayscale: false,
                    plate: -1
                });
                creatingArea = areas.length - 1;
            } else {
                creatingArea = nextArea;
                areas[creatingArea].x1 = imagePos.x;
                areas[creatingArea].y1 = imagePos.y;
                areas[creatingArea].x2 = imagePos.x;
                areas[creatingArea].y2 = imagePos.y;
                areas[creatingArea].grayscale = false;
                areas[creatingArea].plate = -1;
            }
        } else {
            areas.splice(nextArea, 1);
            creatingArea = null;

        }
        draw_fixture();
    });

    selected_fixture_canvas_jq.mousemove(function (event) {
        if (event.button == 0 && isArea(creatingArea)) {
            var canvasPos = selected_fixture_canvas.relMouseCoords(event);
            var imagePos = translateToImageCoords(canvasPos);
            areas[creatingArea].x2 = imagePos.x;
            areas[creatingArea].y2 = imagePos.y;
            draw_fixture();
        }
    });

    selected_fixture_canvas_jq.mouseup( function(event) {
        var minUsableSize = 10000;
        var curArea = creatingArea;
        creatingArea = null;
        if (isArea(curArea)) {
            var area = JSON.parse(JSON.stringify(areas[curArea]));
            area.x1 = Math.min(areas[curArea].x1, areas[curArea].x2);
            area.x2 = Math.max(areas[curArea].x1, areas[curArea].x2);
            area.y1 = Math.min(areas[curArea].y1, areas[curArea].y2);
            area.y2 = Math.max(areas[curArea].y1, areas[curArea].y2);
            areas[curArea] = area;
        }

        for (var i=0; i<areas.length;i++) {
            if (getAreaSize(i) < minUsableSize) {
                areas.splice(i, 1);
                if (i < curArea)
                    curArea--;
                else if (i == curArea)
                    curArea = -1;
                i--;

            }
        }

        if (curArea >= 0 && hasGrayScale() == false)
            testAsGrayScale(curArea);

        setPlateIndices();
        draw_fixture();
     });

}

function isArea(index) {
    return index != null && index >= 0 && index < areas.length;
}

function getAreaSize(plate) {

    if (isInt(plate)) {
        if (plate >=0 && plate < areas.length)
            plate = areas[plate];
        else
            plate = null;
    }

    if (plate)
        return Math.abs(plate.x2 - plate.x1) * Math.abs(plate.y2 - plate.y1);
    return -1;
}

function getAreaCenter(plate) {

    if (isInt(plate)) {
        if (plate >=0 && plate < areas.length)
            plate = areas[plate];
        else
            plate = null;
    }

    if (plate)
        return {
            x: (plate.x1 + plate.x2) / 2,
            y: (plate.y1 + plate.y2) / 2
        }
    else
        return {x: selected_fixture_canvas.width/2,
                y: selected_fixture_canvas.height/2};
}

function isInt(value) {
  return !isNaN(value) &&
         parseInt(Number(value)) == value &&
         !isNaN(parseInt(value, 10));
}

function setPlateIndices() {
    areas.sort(function(a, b) {
        if (a.grayscale)
            return -1;
        else if (b.grayscale)
            return 1;

        if (a.y2 < b.y1)
            return -1;
        else if (b.y2 < a.y1)
            return 1;
        else if (a.x2 < b.x1)
            return -1;
        else if (b.x2 < a.x1)
            return 1;

        var aCenter = getAreaCenter(a);
        var bCenter = getAreaCenter(b);

        return aCenter.y < bCenter.y ? -1 : 1;
    });
    var len = areas.length;
    var plateIndex = 1;
    for (var i=0; i<len; i++) {
        if (areas[i].grayscale !== true && getAreaSize(i) > 0) {
            areas[i].plate = plateIndex;
            plateIndex++;
        }
    }
}

function clearAreas() {
    areas = [];
}

function getAreaByPoint(point) {
    for (var len = areas.length, i=0; i<len; i++) {
        if (isPointInArea(point, areas[i])) {
            return i;
        }
    }
    return -1;
}

function isPointInArea(point, area) {
    return area.x1 < point.x && area.x2 > point.x && area.y1 < point.y && area.y2 > point.y;
}

function hasGrayScale() {
    for (var len=areas.length, i=0;i<len;i++) {
        if (areas[i].grayscale)
            return true;
    }
    return false;
}

function removeGrayScale() {
    for (var len=areas.length, i=0;i<len;i++) {
        if (areas[i].grayscale)
            areas.grayscale = false;
            break;
    }
}

function testAsGrayScale(plate) {
    if (isInt(plate)) {
        if (isArea(plate))
            plate = areas[plate];
        else
            plate = null;
    }

    if (plate) {
        var grayscale_name = $(grayscale_id).val();
        $.ajax({
            url: "?grayscale=1&fixture=" + fixture_name + "&grayscale_name=" + grayscale_name,
            method: "POST",
            data: plate,
            success: function (data) {
                console.log(data);
                if (data.source_values && data.source_values.length > 0)
                    plate.grayscale = true;
                else
                    plate.grayscale = false;
                draw_fixture();
            },
            error: function (data) {
                console.log(data);
                context_warning = "Error occured detecting grayscale";
                draw_fixture();
            }

        });
    }
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
    fixture_name = name;
    $(fixture_name_id).text(get_fixture_as_name(name));
    clearAreas();
    selected_fixture_canvas_jq.focus();
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

    var canvas =  selected_fixture_canvas;
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
        var canvasCenter = getAreaCenter(null);
        context.font = '20pt Calibri';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillStyle = 'red';
        context.fillText(context_warning, canvasCenter.x, canvasCenter.y);
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

    if (getAreaSize(plate) <= 0)
        return;
    context.beginPath();
    context.rect(plate.x1 * scale, plate.y1 * scale, (plate.x2 - plate.x1) * scale, (plate.y2 - plate.y1) * scale);
    context.fillStyle = "rgba(0, 255, 0, 0.1)";
    context.fill();
    context.strokeStyle = "green";
    context.lineWidth = 2;
    context.stroke();

    shadow_text(context, plate, "green", "white", plate.grayscale ? "G" : plate.plate)
}

function shadow_text(context, area, text_color, shadow_color, text) {
    var fontSize = Math.min(area.x2 - area.x1, area.y2 - area.y1) * scale * 0.6;
    var center = getAreaCenter(area);

    context.font =  fontSize * 1.1 + 'pt Calibri';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = shadow_color;
    context.fillText(text, center.x * scale, center.y * scale);

    context.font =  fontSize + 'pt Calibri';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = text_color;
    context.fillText(text, center.x * scale, center.y * scale);
}