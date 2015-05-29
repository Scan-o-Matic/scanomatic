
function GetMousePosRelative(event, obj) {
    return {x: event.pageX - obj.offset().left, y: event.pageY - obj.offset().top};
}

function isInt(value) {
  return !isNaN(value) &&
         parseInt(Number(value)) == value &&
         !isNaN(parseInt(value, 10));
}

function InputEnabled(obj, isEnabled) {
    if (isEnabled) {
        obj.removeAttr("disabled");
    } else {
        obj.attr("disabled", true);
    }

}

function unselect(target) {
    target.val("");
}

function get_fixture_as_name(fixture) {
    return fixture.replace(/_/g, " ")
        .replace(/[^ a-zA-Z0-9]/g, "")
        .replace(/^[a-z]/g,
            function ($1) { return $1.toUpperCase();});
}

function get_fixture_from_name(fixture) {
    return fixture.replace(/ /g, "_")
        .replace(/A-Z/g, function ($1) { return $1.toLowerCase();})
        .replace(/[^a-z1-9_]/g,"");
}

function Execute(idOrClass, methodName) {
    $(idOrClass).each(function(i, obj) {
        methodName($(obj));
    });
}