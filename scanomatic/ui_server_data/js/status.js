var okIMG = "/images/yeastOK.png";
var nokIMG = "/images/yeastNOK.png"

function updateStatus(target, statusType, contentFormatter) {
    API.get(`/api/status/${statusType}`)
        .then(response => $(target).html(contentFormatter(response)))
        .catch((reason) => {
            if (reason) {
                $(target).html("<em>Request refused<em>, reason: " + reason);
            } else {
                $(target).html("<em>Error occurred in UI-server processing request</em>");
            }
        });
}

function serverStatusFormatter(data) {
    return `<img src='${data.ResourceCPU ? okIMG : nokIMG}' class='icon'> CPU | <img src='${data.ResourceMem ? okIMG : nokIMG}' class='icon'> Memory | Uptime: ${data.ServerUpTime}`;
}

function scannerStatusFormatter(response) {
    const data = response.scanners;
    if (data.length === 0) {
        return '<em>No scanners are connected according to Scan-o-Matic. ' +
            'If this feels wrong, verify your power-manager settings and that the power-manager is reachable.</em>';
    }
    let ret = '';
    for (let i=0; i<data.length; i++) {
        ret += "<div class='scanner'><h3>" + data[i].scanner_name + "</h3>" +
            "<code>" + (data[i].power ? "Has power" : "Is offline") + "</code>" +
            "<p class=''>" + getOwnerName(data[i]) + "</p>" +
            "</div>";
    }

    return ret;
}

function getOwnerName(data) {
    if (data.owner) {
        if (data.email)
            return 'Owner: ' + data.email;
        else if (data.owner.content_model && data.owner.content_model.email)
            return 'Owner: ' + data.owner.content_model.email;
        else
            return 'Owner unknown';
    }
    return 'Free to use';
}

function queueStatusFormatter(response) {
    const data = response.queue;
    if (data.length === 0) {
        return '<em>No jobs in queue... if it feels like the job disappeared, it is because it may take a few seconds before it pops up below.</em>';
    }
    let ret = '';

    for (let i = 0; i < data.length; i += 1) {
        ret += queueItemAsHTML(data[i]);
    }
    return ret;
}

function jobsStatusFormatter(response) {
    const data = response.jobs;
    if (data.length == 0)
        return '<em>No jobs running</em>';

    let ret = '';

    for (let i = 0; i < data.length; i += 1) {
        ret += jobStatusAsHTML(data[i]);
    }
    return ret;

}

function jobStatusAsHTML(job) {
    ret = "<div class=job title='ETA: "
        + (job.progress > 0.01 ? ((job.runTime / job.progress - job.runTime) / 60).toFixed(1) : "???") +
        "min'><input type='hidden' class='id' value='" + job.id + "'><code>"
        + job.type + "</code>&nbsp;<code>"
        + (job.running ? "Running" : "Not running") + "</code>";

    if (job.stopping)
        ret += '&nbsp;<code>Stopping</code>';

    if (job.paused)
        ret += '&nbsp;<code>Paused</code>';

    if (job.progress != -1)
        ret += ' | <code>' + (job.progress * 100).toFixed(1) + '% progress</code>&nbsp;'
    else
        ret += ' | <code>Progress unknown</code>&nbsp;';

    ret += job.label;

    if (job.log_file) {

        ret += "<span class='log-link'><a href='" + job.log_file  + "'><img src='/images/log_icon.png' height=24px></a></span>"
    }

    ret += "<button type='button' class='stop-button' onclick='stopDialogue(this);'></button>";
    return ret + "</div>";
}

function shortHash(job) {
    return ` (${job.id.substring(job.id.length - 8)})`;
}

function queueItemAsHTML(job) {

    ret = "<div class='job'><input type='hidden' class='id' value='" + job.id + "'><code>"
        + job.type + "</code>&nbsp;<code>" + job.status + "</code>&nbsp;";

    if (job.type == "Scan") {
        ret += job.content_model.project_name;
    } else if (job.type == "Compile") {
        arr = job.content_model.path.split("/");
        ret += arr[arr.length - 1];
    } else if (job.type == "Analysis") {
        arr = job.content_model.compilation.split("/");
        ret += arr[arr.length - 2].replace("_", " ") + " -> " + job.content_model.output_directory.replace("_", " ");
    } else if (job.type == "Features") {
        arr = job.content_model.analysis_directory.split("/");
        ret += arr[arr.length - 2].replace("_", " ") + ": " + arr[arr.length - 1].replace("_", " ");
    }
    ret += shortHash(job);

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
                        url: "/api/job/" + job_id + "/stop",
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
