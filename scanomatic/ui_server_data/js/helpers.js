
class API {
    static get(url) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'GET',
            success: resolve,
            error: jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
        }));
    }

    static postFormData(url, formData) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'POST',
            contentType: false,
            enctype: 'multipart/form-data',
            data: formData,
            processData: false,
            success: resolve,
            error: jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
        }));
    }

    static postJSON(url, json) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'POST',
            data: JSON.stringify(json),
            contentType: 'application/json',
        })
            .then(
                resolve,
                jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
            ));
    }
}

function get_path_suggestions(input, isDirectory, suffix, suffix_pattern, callback, prefix, checkHasAnalysis) {

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

            if (suffix_pattern != null) {
                filtered = [];
                filtered_is_directories = [];
                i = data.suggestions.length;
                while (i--) {
                    if (data.suggestion_is_directories[i] || suffix_pattern.test(data.suggestions[i])) {
                        filtered.push(data.suggestions[i]);
                        filtered_is_directories.push(data.suggestion_is_directories[i]);
                    }
                }
                data.suggestions = filtered;
                data.suggestion_is_directories = filtered_is_directories;
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

function ArrMap(arr, lambda_func) {
    new_arr = [];
    for (i=0; i<arr.length; i++) {
        new_arr[i] = lambda_func(arr[i]);
    }
    return new_arr;
}
