var current_fixture_id;
var new_fixture_data_id;

function get_fixture_as_name(fixture) {
    return fixture.replace("_", " ")
        .replace(/^[a-z]/g,
            function ($1) { return $1.toUpperCase();});
}

function unselect(target) {
    target.val("");
    //target.("select option:selected" ).each(function() {this.prop("selected", false);})
}

function get_fixtures() {
    var options = $(current_fixture_id);
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
    $(new_fixture_data_id).show()
}

function get_fixture() {
    var options = $(current_fixture_id);
    $(new_fixture_data_id).hide();

}