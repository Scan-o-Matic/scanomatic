var current_fixture_id;
var new_fixture_data_id;
var new_fixture_detect_id;
var new_fixture_image_id;
var new_fixture_markers_id;
var selected_fixture_div_id;
var fixture_name_id;
var selected_fixture_canvas_id;
var new_fixture_name;

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
    var options = $(current_fixture_id)
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
        set_fixture_markers(data);
    }

});
    load_fixture($(new_fixture_name).val(), $(new_fixture_image_id)[0].files[0]);
}

function update_fixture_name() {
    $(fixture_name_id).text(get_fixture_as_name($(new_fixture_name).val()));
}

function load_fixture(name, img_data) {
    $(new_fixture_data_id).hide();
    $(fixture_name_id).text(get_fixture_as_name(name));
    $(selected_fixture_div_id).show();
    var ctx = $(selected_fixture_canvas_id)[0].getContext('2d');
    if (img_data) {
        var img = new Image;
        img.onload = function() {ctx.drawImage(img, 0,0);}
        img.src = URL.createObjectURL(img_data);
    }
}

function set_fixture_markers(data) {
    console.log(data);
}