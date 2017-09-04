
function getUrlParameter(sParam) {
    var sPageUrl = decodeURIComponent(window.location.search.substring(1));
    var sUrlVariables = sPageUrl.split('&');
    var sParameterName, i;

    for (i = 0; i < sUrlVariables.length; i++) {
        sParameterName = sUrlVariables[i].split("=");

        if (sParameterName[0] === sParam) {
            return sParameterName[1] === undefined ? true : sParameterName[1];
        }
    }
    return null;
};

function FillPlate() {
    //32x48
    var count = 0;
    var plate = d3.range(48).map(function () {
        return d3.range(32).map(function () {
            count += 1;
            return count;
        });
    });
    return plate;
}

function getLastSegmentOfPath(path) {
    var parts = path.split("/");
    var lastPart = parts.pop();
    if (lastPart === "")
        lastPart = parts.pop();
    return lastPart;
}

function getExtentFromMultipleArrs() {
    if (!arguments.length) return null;
    var extremeValues = [];
    for (var i = 0; i < arguments.length; i++) {
        extremeValues.push(d3.max(arguments[i]));
        extremeValues.push(d3.min(arguments[i]));
    }
    var ext = d3.extent(extremeValues);
    return ext;
}

function getBaseLog(base, value) {
    return Math.log(value) / Math.log(base);
}
