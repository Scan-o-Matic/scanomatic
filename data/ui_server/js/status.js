var okIMG = "/images/yeastOK.png";
var nokIMG = "/images/yeastNOK.png"

function updateStatus(target, status_type, content_formatter) {
    $.ajax({
        url: "/status/" + status_type,
        method: "GET",
        success: function (data) {
            if (data.success) {
                $(target).html(content_formatter(data.data));
            } else {
                $(target).html("<em>Request refused<em>, reason: " + data.reason);
            }
        },
        error: function (data) {
            $(target).html("<em>Error occurred in UI-server processing request</em>");
        }
    });
}

function serverStatusFormatter(data) {
    return "<img src='" + (data.ResourceCPU ? okIMG : nokIMG) + "'> CPU | <img src='" +
        (data.ResourceMem ? okIMG : nokIMG) + "'> Memory | Uptime: " + data.ServerUpTime;
}

function scannerStatusFormatter(data) {
    ret = "";

    for (var i=0; i<data.length; i++) {
        ret += "<div class='scanner'><h3>" + data[i].scanner_name + "</h3>" +
            "<code>" + (data[i].power ? "Has power" : "Is offline") + "</code>" +
            "<p class=''>" + (data[i].owner ? (data[i].email != "" ? "Owner: " + data[i].email : "Occupied") :"Free to use") + "</p>" +
            "</div>";
    }

    return ret;
}

function queueStatusFormatter(data) {
    if (data.length == 0)
        return "<em>No jobs in queue... if it feels like the job disappeared, it is because it may take a few seconds before it pops up below.</em>";

    ret = "";

    for (var i=0;i<data.length;i++)
        ret += jobAsHTML(data[i]);

    return ret;
}

function jobsStatusFormatter(data) {
    if (data.length == 0)
        return "<em>No jobs running</em>";

    ret = "";

    for (var i=0;i<data.length;i++)
        ret += jobStatusAsHTML(data[i]);

    return ret;

}

function jobStatusAsHTML(job) {
    ret = "<div class=job><code>" + job.type + "</code><img src='" + (job.running ? okIMG : nokIMG) + "'>Running | ";

    if (job.stopping)
        ret += "<code>Stopping</code> | ";

    if (job.paused)
        ret += "<code>Paused</code> | ";

    if (job.progress != -1)
        ret += "<code>" + (job.progress * 100).toFixed(1) + "% progress</code>&nbsp;"
    else
        ret += "<code>Progress unknown</code>&nbsp;";

    ret += job.label;

    return ret + "</div>";
}

function jobAsHTML(job) {
    ret = "<div class='job'><code>" + job.type + "</code>\t<code>" + job.status + "</code>\t";
    if (job.type == "Scan")
        ret += job.content_model.project_name;
    else if (job.type == "Compile")
        ret += job.content_model.project_name;
    else if (job.type == "Analysis")
        ret += job.content_model.compilation;
    else if (job.type == "Features")
        ret += job.content_model.path;
    else
        ret += job.id;
    return ret + "</div>";
}