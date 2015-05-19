
function get_fixture_as_name(fixture) {
    return fixture.replace("_", " ")
        .replace(/^[a-z]/g,
            function ($1) { return $1.toUpperCase();});
}

function unselect(target) {
    target.val("");
    //target.("select option:selected" ).each(function() {this.prop("selected", false);})
}

function get_fixtures(target) {
    var options = $(target);
    $.get("/fixtures?names=1", function(data, status) {
        $.each(data.split(","), function() {
            options.append($("<option />").val(this).text(get_fixture_as_name(this)));
        })
        unselect(options);
    })
}

function add_fixture(target) {
    var options = $(target)
    unselect(options);
}

function get_fixture(target) {
    var options = $(target);
    console.log(options.val());
}