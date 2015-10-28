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
    return "<img src='" + (data.ResourceCPU ? okIMG : nokIMG) + "' class='icon'> CPU | <img src='" +
        (data.ResourceMem ? okIMG : nokIMG) + "' class='icon'> Memory | Uptime: " + data.ServerUpTime;
}

function scannerStatusFormatter(data) {
    ret = "";

    for (var i=0; i<data.length; i++) {
        ret += "<div class='scanner'><h3>" + data[i].scanner_name + "</h3>" +
            "<code>" + (data[i].power ? "Has power" : "Is offline") + "</code>" +
            "<p class=''>" + getOwnerName(data[i].owner) + "</p>" +
            "</div>";
    }

    if (data.length == 0) {
        ret = "<em>No scanners are connected according to Scan-o-Matic. " +
            "If this feels wrong, verify your power-manager settings and that the power-manager is reachable.</em>";
    }

    return ret;
}

function getOwnerName(owner) {
    if (owner) {
        if (owner.content_model && owner.content_model.email)
            return "Owner: " + owner.content_model.email;
        else
            return "Owner unknown";
    }
    return "Free to use";
}

function queueStatusFormatter(data) {
    if (data.length == 0)
        return "<em>No jobs in queue... if it feels like the job disappeared, it is because it may take a few seconds before it pops up below.</em>";

    ret = "";

    for (var i=0;i<data.length;i++)
        ret += queueItemAsHTML(data[i]);

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
    ret = "<div class=job><input type='hidden' class='id' value='" + job.id + "'><code>"
        + job.type + "</code>&nbsp;<code>"
        + (job.running ? "Running" : "Not running") + "</code>";

    if (job.stopping)
        ret += "&nbsp;<code>Stopping</code>";

    if (job.paused)
        ret += "&nbsp;<code>Paused</code>";

    if (job.progress != -1)
        ret += " | <code>" + (job.progress * 100).toFixed(1) + "% progress</code>&nbsp;"
    else
        ret += " | <code>Progress unknown</code>&nbsp;";

    ret += job.label;

    ret += "<button type='button' class='stop-button' onclick='stopDialogue(this);'></button>";
    return ret + "</div>";
}

function queueItemAsHTML(job) {

    ret = "<div class='job'><input type='hidden' class='id' value='" + job.id + "'><code>"
        + job.type + "</code>&nbsp;<code>" + job.status + "</code>&nbsp;";

    if (job.type == "Scan")
        ret += job.content_model.project_name;
    else if (job.type == "Compile")
        ret += job.content_model.path;
    else if (job.type == "Analysis")
        ret += job.content_model.compilation;
    else if (job.type == "Features")
        ret += job.content_model.analysis_directory;
    else
        ret += job.id;

    ret += "<button type='button' class='stop-button' onclick='stopDialogue(this);'></button>";
    return ret + "</div>";
}

function stopDialogue(button) {

    button = $(button);
    var title = "Terminate job";
    var job_id = button.siblings(".id").first().val();
    var body_header = "Are you sure?";
    var body = "This will terminate the job '', click 'Yes' to proceed.";

    InputEnabled(button, false);

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {

                        button = null;

                        $.ajax({
                        url: "/job/" + job_id + "/stop",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Accepted", "It may take a little while before stop is executed, so please be patient");
                            } else {
                                Dialogue("Not allowed", data.reason);
                            }
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (button)
                InputEnabled(button, true);
        });

}