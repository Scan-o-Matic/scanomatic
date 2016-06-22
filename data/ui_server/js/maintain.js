function animate_button_waiting(button) {
    InputEnabled(button, false);
    /*
    var txt = button.html()
    while (IsDisabled(button)) {

    }

    */
}

function disable_all_buttons() {
    $(":button").hide();
}

function enable_all_buttons() {
    $(":button").show();
}

function server_reboot(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "Server";
    var body_header = "Are you sure?";
    var body = "This will reboot the server and may cause problems if any jobs are active, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/server/reboot",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Completed", "The server is now rebooted.");
                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function server_shutdown(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "Server";
    var body_header = "Are you sure?";
    var body = "This will shutdown the server and may cause problems if any jobs are active, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/server/shutdown",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Completed", "The server is now shutdown.");
                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function server_kill(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "Server";
    var body_header = "Are you sure?";
    var body = "This will kill the server process and any processes named too similarly. " +
     "It may cause problems if any jobs are active and worse issues if you are unlucky... click 'Yes' to proceed ";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/server/kill",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Completed", "The server is now killed.");
                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function server_launch(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "Server";
    var body_header = "Launch?";
    var body = "This will attempt to launch the server, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/server/launch",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Completed", "The server is up and running.");
                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function app_reboot(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "App";
    var body_header = "Are you sure?";
    var body = "This will reboot the app, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/app/reboot",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Completed", "The app is now rebooted.");
                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}


function app_shutdown(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "App";
    var body_header = "Are you sure?";
    var body = "This will shutdown the app but leave the server running, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/app/shutdown",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {
                                Dialogue("Completed", "The app is now shutdown.");
                                window.close();
                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function app_server_reboot(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "App & Server";
    var body_header = "Are you sure?";
    var body = "This will reboot the app & the server, this may cause issues if jobs are running, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/server/shutdown",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {

                                $.ajax({
                                url: "/api/app/reboot",
                                method: "GET",
                                success: function (data) {
                                    if (data.success) {
                                        Dialogue("Completed", "The app & server are now rebooted.");
                                    } else {
                                        Dialogue("Failed", data.reason);
                                    }
                                    enable_all_buttons();
                                },
                                error: function (data) {
                                    Dialogue("Error", data.reason);
                                    enable_all_buttons();
                                }});

                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function app_server_shutdown(button) {

    disable_all_buttons();
    var reset_button = true;
    var title = "App & Server";
    var body_header = "Are you sure?";
    var body = "This will shutdown the app & the server, this may cause issues if jobs are running, click 'Yes' to proceed.";

    $('<div class=\'dialog\'></div>').appendTo("body")
        .prop("title", title)
        .html("<div><h3>" + body_header + "</h3>" + body + "</div>")
        .dialog({modal: true,
                 buttons: {
                    Yes: function() {
                        reset_button = false;
                        $.ajax({
                        url: "/api/server/shutdown",
                        method: "GET",
                        success: function (data) {
                            if (data.success) {

                                $.ajax({
                                url: "/api/app/shutdown",
                                method: "GET",
                                success: function (data) {
                                    if (data.success) {
                                        Dialogue("Completed", "The app & server are now shutdown.");
                                    } else {
                                        Dialogue("Failed", data.reason);
                                    }
                                    enable_all_buttons();
                                },
                                error: function (data) {
                                    Dialogue("Error", data.reason);
                                    enable_all_buttons();
                                }});

                            } else {
                                Dialogue("Failed", data.reason);
                            }
                            enable_all_buttons();
                        },
                        error: function (data) {
                            Dialogue("Error", data.reason);
                            enable_all_buttons();
                        }});
                        $(this).dialog("close");
                    },
                    No: function() {
                        $(this).dialog("close");
                    }
                 }
        }).on('dialogclose', function(event) {
            if (reset_button  )
                enable_all_buttons();
        });

}

function setVersionInformation() {
    $.ajax({
    url: "/api/app/version",
    method: 'GET',
    success: function(data) {

        if (data["source_information"] && data["source_information"]["branch"])
            branch = ", " + data["source_information"]["branch"];
        else
            branch = "";
        $("#version-info").html("Scan-o-Matic " + data["version"] + branch);
    },
    error: function(data) {
        $("#version-info").html("Error requesting version of Scan-o-Matic, this should not happen... reboot?");
    }
    });
}