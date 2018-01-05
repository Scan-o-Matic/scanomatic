var current_fixture_id;
var new_fixture_data_id;
var new_fixture_detect_id;
var new_fixture_image_id;
var new_fixture_markers_id;
var selected_fixture_div_id;
var fixture_name_id;
var selected_fixture_canvas_id;
var new_fixture_name;
var save_fixture_action_id;
var save_fixture_button;
var remove_fixture_id;
var grayscale_type_id;


var context_warning = "";
var fixture_image = null;
var fixture_name = null;
var markers = null;
var scale = 1;
var areas = [];
var creatingArea = null;
var selected_fixture_canvas_jq;
var selected_fixture_canvas;
var grayscale_graph = null;
var img_width = 0;
var img_height = 0;

$(document.documentElement).mouseup(function(event) {
    if (creatingArea != null && selected_fixture_canvas_jq != null)
        mouseUpFunction(event);
});

function translateToImageCoords(coords) {
    var imageCoords = JSON.parse(JSON.stringify(coords));
    imageCoords.x = clamp(imageCoords.x, 0, img_width) / scale;
    imageCoords.y = clamp(imageCoords.y, 0, img_height) / scale;
    return imageCoords;
}

function set_canvas() {

    selected_fixture_canvas_jq = $(selected_fixture_canvas_id);
    selected_fixture_canvas_jq.attr("tabindex", "0");
    selected_fixture_canvas = selected_fixture_canvas_jq[0];

    selected_fixture_canvas_jq.mousedown(function (event) {

        if (context_warning) {
            context_warning = null;
            return;
        }

        var canvasPos = GetMousePosRelative(event, selected_fixture_canvas_jq);
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
                if (areas[creatingArea].grayscale)
                    grayscale_graph = null;
                areas[creatingArea].x1 = imagePos.x;
                areas[creatingArea].y1 = imagePos.y;
                areas[creatingArea].x2 = imagePos.x;
                areas[creatingArea].y2 = imagePos.y;
                areas[creatingArea].grayscale = false;
                areas[creatingArea].plate = -1;
                InputEnabled($(grayscale_type_id), true);
            }
        } else {
            if (areas[nextArea] && areas[nextArea].grayscale) {
                grayscale_graph = null;
                InputEnabled($(grayscale_type_id), true);
            }
            areas.splice(nextArea, 1);
            creatingArea = null;

        }
        draw_fixture();
    });

    selected_fixture_canvas_jq.mousemove(function (event) {
        var canvasPos = GetMousePosRelative(event, selected_fixture_canvas_jq);
        var imagePos = translateToImageCoords(canvasPos);

        if (event.button == 0 && isArea(creatingArea)) {

            areas[creatingArea].x2 = imagePos.x;
            areas[creatingArea].y2 = imagePos.y;
            draw_fixture();
        }

        draw_hover_slice(imagePos);
    });

    selected_fixture_canvas_jq.mouseup(mouseUpFunction );

}

function mouseUpFunction(event) {
        var minUsableSize = 10000;
        var curArea = creatingArea;
        creatingArea = null;
        if (isArea(curArea)) {
            var area = JSON.parse(JSON.stringify(areas[curArea]));
            var imagePos = translateToImageCoords({x: img_width, y: img_height});
            area.x1 = Math.max(Math.min(areas[curArea].x1, areas[curArea].x2, imagePos.x), 0);
            area.x2 = Math.min(Math.max(areas[curArea].x1, areas[curArea].x2), imagePos.x);

            area.y1 = Math.max(Math.min(areas[curArea].y1, areas[curArea].y2, imagePos.y), 0);
            area.y2 = Math.min(Math.max(areas[curArea].y1, areas[curArea].y2), imagePos.y);
            areas[curArea] = area;
        }

        for (var i=0; i<areas.length;i++) {
            if (getAreaSize(i) < minUsableSize) {
                if (area[i] && area[i].grayscale) {
                    grayscale_graph = null;
                }
                areas.splice(i, 1);
                if (i < curArea)
                    curArea--;
                else if (i == curArea)
                    curArea = -1;
                i--;

            }
        }

        if (curArea >= 0) {
            if (hasGrayScale())
                areas[curArea].plate = 0;
            else
                testAsGrayScale(curArea);
        }
        setPlateIndices();
        draw_fixture();
}

function isArea(index) {
    return index != null &&  index != undefined && index >= 0 && index < areas.length;
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
            return 1;
        else if (b.x2 < a.x1)
            return -1;

        var aCenter = getAreaCenter(a);
        var bCenter = getAreaCenter(b);

        return aCenter.y < bCenter.y ? -1 : 1;
    });
    var len = areas.length;
    var plateIndex = 1;
    for (var i=0; i<len; i++) {
        if (areas[i].grayscale !== true && areas[i].plate >= 0 && getAreaSize(i) > 0) {
            areas[i].plate = plateIndex;
            plateIndex++;
        }
    }
}

function clearAreas() {
    areas = [];
    grayscale_graph = null;
    context_warning = "";
    InputEnabled($(grayscale_type_id), true);
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

function testAsGrayScale(plate) {
    if (isInt(plate)) {
        if (isArea(plate))
            plate = areas[plate];
        else
            plate = null;
    }

    if (plate) {
        var grayscale_name = GetSelectedGrayscale();
        $.ajax({
            url: "/api/data/grayscale/fixture/" + fixture_name + "?grayscale_name=" + grayscale_name,
            method: "POST",
            data: plate,
            success: function (data) {
                console.log(data);
                if (data.grayscale && hasGrayScale() === false)  {
                    plate.grayscale = true;
                    grayscale_graph = GetLinePlot(data.target_values, data.source_values,
                        "Grayscale", "Targets", "Measured values");
                    InputEnabled($(grayscale_type_id), false);
                } else {
                    if (!hasGrayScale() && data.reason)
                        grayscale_graph = GetLinePlot([], [], data.reason, "Targets", "Measured values");
                    plate.grayscale = false;
                    plate.plate = 0;
                    InputEnabled($(grayscale_type_id), true);
                    setPlateIndices();
                }
                draw_fixture();
            },
            error: function (data) {
                console.log(data);
                context_warning = "Error occured detecting grayscale";
                setPlateIndices();
                draw_fixture();
                InputEnabled($(grayscale_type_id), true);
            }

        });
    }
}

function SetAllowDetect() {

    var disallow = $(new_fixture_image_id).val() == "" || $(new_fixture_name).val() == "";
    InputEnabled($(new_fixture_detect_id), !disallow);
}

function get_fixtures() {
    var options = $(current_fixture_id);
    options.empty();
    $.get("/api/data/fixture/names", function(data, status) {
        $.each(data.fixtures, function() {
            options.append($("<option />").val(this).text(get_fixture_as_name(this)));
        });
        unselect(options);
    });
    $(new_fixture_data_id).hide();
    $(selected_fixture_div_id).hide();
}

function add_fixture() {
    var options = $(current_fixture_id);
    unselect(options);
    unselect($(new_fixture_image_id));
    $(save_fixture_action_id).val("create");
    SetAllowDetect();
    $(new_fixture_detect_id).val("Detect");
    $(new_fixture_data_id).show();
    $(selected_fixture_div_id).hide();
}

function get_fixture() {
    var options = $(current_fixture_id);
    $(new_fixture_data_id).hide();
    $(save_fixture_action_id).val("update");
    load_fixture(options.val());
    load_fixture_image(get_fixture_from_name(options.val()));
    $.ajax({
        url: '/api/data/fixture/get/' + options.val(),
        type: "GET",
        success: function(data) {
            if (data.success) {
                areas.splice(0);
                for (var i=0, l=data.plates.length;i<l;i++) {
                    data.plates[i].grayscale = false;
                    data.plates[i].plate = data.plates[i].index;
                    areas.push(data.plates[i]);
                }
                data.grayscale.grayscale = true;
                data.grayscale.plate = -1;
                SetSelectedGrayscale(data.grayscale.name);
                InputEnabled($(grayscale_type_id), false);

                areas.push(data.grayscale)
                markers = data.markers;

            } else if (data.reason)
                context_warning = data.reason;
            else
                context_warning = "Unknown error retrieving fixture";
            draw_fixture();
        }
        });
}

function detect_markers() {

    var formData = new FormData();
    formData.append("markers", $(new_fixture_markers_id).val());
    formData.append("image", $(new_fixture_image_id)[0].files[0]);
    InputEnabled($(new_fixture_detect_id), false);
    var button = $(save_fixture_button);
    InputEnabled(button, false);
    $(new_fixture_detect_id).val("...");
    $.ajax({
        url: '/api/data/markers/detect/' + $(new_fixture_name).val(),
        type: 'POST',
        contentType: false,
        enctype: 'multipart/form-data',
        data: formData,
        processData: false,
        success: function (data) {
            var new_image = $(new_fixture_image_id);
            load_fixture($(new_fixture_name).val());

            if (data.image && data.markers) {
                context_warning = ""
                $(new_fixture_data_id).hide();
            } else {
                context_warning = "Name or image refused";
            }
             load_fixture_image(data.image);
             set_fixture_markers(data);
             InputEnabled(button, true);
        },
        error: function (data) {
            context_warning = "Marker detection failed";
            markers = null;
            load_fixture($(new_fixture_name).val());
            draw_fixture();
        }});
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
        img.src = "/api/data/fixture/image/get/" + image_name + "?rnd=" + Math.random();

    } else {
        fixture_image = null;
    }
}

function set_fixture_markers(data) {
    markers = data.markers;
    if (markers.length ==0) {
        markers = null;
        context_warning = "No markers were detected!";
        $(new_fixture_data_id).show();
    }
    draw_fixture();
}

function SaveFixture() {
    var button = $(save_fixture_button);
    InputEnabled(button, false);
    payload = {
        markers: markers,
        grayscale_name: GetSelectedGrayscale(),
        areas: areas};
    $.ajax({
        url:"/api/data/fixture/set/" + fixture_name,
        data: JSON.stringify(payload, null, '\t'),
        contentType: 'application/json;charset=UTF-8',
        dataType: "json",
        processData: false,
        method: "POST",
        success: function(data) {
            if (data.success) {
                $(selected_fixture_div_id).hide();
                Dialogue('Fixture', 'Fixture "' + fixture_name + '" saved', '','?');
            } else {
                if (data.reason)
                    context_warning = "Save refused: " + data.reason;
                else
                    context_warning = "Save refused";
                InputEnabled(button, true);
            }
            draw_fixture();
        },
        error: function(data) {
            context_warning = "Crash while trying to save";
            draw_fixture();
            InputEnabled(button, true);
        }
    });
}

function RemoveFixture() {

    $('<div class=\'dialog\'></div>').appendTo("body")
                    .html('<div><h3>Are you sure you want to remove \'' + fixture_name + '\'?')
                    .dialog({
                        modal: true,
                        title: "Remove",
                        zIndex: 10000,
                        autoOpen: true,
                        width: 'auto',
                        resizable: false,
                        buttons: {
                            Yes: function() {

                                payload = {};

                                $.ajax({
                                    url: "/api/data/fixture/remove/" + fixture_name,
                                    method: "GET",
                                    success: function(data) {
                                        if (data.success) {
                                            $(selected_fixture_div_id).hide();
                                            Dialogue('Fixture', 'Fixture "' + fixture_name + '" has been removed',
                                                '(A backup is always stored in the fixture config folder)', '?');

                                        } else {
                                            console.log(data);
                                            if (data.reason)
                                                context_warning = data.reason;
                                            else
                                                context_warning = "Unknown removal issue";

                                            draw_fixture();
                                        }
                                    },
                                    error: function(data) {
                                        context_warning = "Crash while removing";
                                        draw_fixture();
                                    }
                                });

                                $(this).dialog("close");
                            },
                            No: function() {
                                $(this).dialog("close");
                            }
                        },
                        close: function(event, ui) {
                            $(this).remove();
                        }
                    });


}

function draw_fixture() {

    var canvas =  selected_fixture_canvas;
    var context = canvas.getContext('2d');

    context.clearRect(0, 0, canvas.width, canvas.height);

    if (fixture_image) {
        scale = get_updated_scale(canvas, fixture_image);
        img_width = fixture_image.width * scale;
        img_height = fixture_image.height * scale;
        context.drawImage(fixture_image, 0, 0, img_width, img_height);
    } else {
        img_width = 0;
        img_height = 0;
    }

    if (markers) {
        var radius = 140 * scale;
        var marker_scale = 1;
        for (var len = markers.length, i=0; i<len;i++)
            draw_marker(context, markers[i][0] * scale * marker_scale,
                        markers[i][1] * scale * marker_scale, radius, "cyan", 3);
    }

    if (areas) {
        for (var i=0; i<areas.length; i++)
            draw_plate(context, areas[i]);
    }

    if (grayscale_graph) {
        var graph_width = (canvas.width - img_width);
        context.drawImage(grayscale_graph, img_width, 0, graph_width, graph_width);
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

    context.beginPath();
    context.moveTo(centerX - 0.5 * radius, centerY - 0.5 * radius);
    context.lineTo(centerX + 0.5 * radius, centerY + 0.5 * radius);
    context.moveTo(centerX + 0.5 * radius, centerY - 0.5 * radius);
    context.lineTo(centerX - 0.5 * radius, centerY + 0.5 * radius);
    context.lineWidth = 0.5 * lineWidth;
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

    shadow_text(context, plate, "green", "white", plate.grayscale ? "G" : plate.plate < 0 ? "?" : plate.plate)
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

function draw_hover_slice(image_coords) {

    if (fixture_image) {

        canvas = selected_fixture_canvas;
        context = canvas.getContext('2d');

        img_half_size = 90;
        preview_size = 180
        cx = canvas.width - preview_size - 10;
        cw = preview_size;
        cy = canvas.height - cw - 10;
        ch = cw;

        context.clearRect(cx, cy, ch, cw);

        iw = 2 * img_half_size + 1;
        ix = Math.max(image_coords.x - img_half_size, 0);

        ih = 2 * img_half_size + 1;
        iy = Math.max(image_coords.y - img_half_size, 0);

        context.drawImage(fixture_image, ix, iy, iw, ih, cx, cy, cw, ch);
    }
}
