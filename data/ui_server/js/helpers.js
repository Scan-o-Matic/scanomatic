function get_path_suggestions(input, isDirectory, suffix, callback, prefix, checkHasAnalysis) {

    if (suffix == undefined)
        suffix = "";

    if (prefix != undefined) {
        url = prefix.replace(/^\/?|\/?$/, "") + "/" + $(input).val().replace(/^\/?|\/?$/, "");
    } else {
        url = $(input).val().replace(/^\/?|\/?$/, "");
    }

    if (url == "" || url == undefined) {
        url = "/api/tools/path";
    } else {
        url = "/api/tools/path/" +  url;
    }

    $.get(url + "?suffix=" + suffix +
        "&isDirectory=" + (isDirectory ? 1 : 0) +
        "&checkHasAnalysis=" + (checkHasAnalysis ? 1 : 0),
        function(data, status) {
            var val = $(input).val();
            if (prefix) {
                start_index = ("root/" + prefix.replace(/^\/?|\/?$/, "")).length;
                for(i=0;i<data.suggestions.length;i++){
                    data.suggestions[i] = data.suggestions[i].substring(start_index, data.suggestions[i].length);
                }
            }
            $(input).autocomplete({source: data.suggestions});
            if (prefix == undefined && (val == "" || (data.path == "root/" && val.length < data.path.length))) {
                $(input).val(data.path);
            }

            callback(data, status);
    });
}

function clamp(value, min, max) {
    return value == null ? min : (isNaN(value) ? max : Math.max(Math.min(value, max), min));
}

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

function IsDisabled(obj) {
    return obj.attr("disabled") == true;
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
        .replace(/[A-Z]/g, function ($1) { return $1.toLowerCase();})
        .replace(/[^a-z1-9_]/g,"") + ".config";
}

function Execute(idOrClass, methodName) {
    $(idOrClass).each(function(i, obj) {
        methodName($(obj));
    });
}

function Dialogue(title, body_header, body, redirect, reactivate_button ) {
    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div>" + (body_header != null ? ("<h3>" + body_header + "</h3>") : "")  + (body != null ? body : "") + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Ok: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (redirect)
                location.href = redirect;
            if (reactivate_button)
                InputEnabled($(reactivate_button), true);
        });

}