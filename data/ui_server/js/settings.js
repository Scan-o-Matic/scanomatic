var pm_type = undefined;

function toggleVisibilities(button, state){
    $(button).parent().nextAll().css('display',state ? '' : "none");
    $("#scanner_section").css('display', $(button).val() != "notInstalled" ? "": "none");
    pm_type = $(button).val();
}

function dynamicallyLimitScanners(button) {
    var numOfScanners = $("#number_of_scanners");
    var maxVal = $(button).val();
    numOfScanners.attr('max', maxVal);
    if (numOfScanners.val() > maxVal)
        numOfScanners.val(maxVal);
}

function UpdateSettings(forceSettings) {
    var action = forceSettings ? "forceUpdate" : "update";
    var data = {
        number_of_scanners: $("#number_of_scanners").val(),
        power_manager: {
            sockets: $("#number_of_sockets").val(),
            type: pm_type,
            host: $("#power_manager_host").val(),
            mac: $("#power_manager_mac").val(),
            name: $("#power_manager_name").val(),
            password: $("#power_manager_password").val(),
        },
        paths: {
            projects_root: $("#projects_root").val()
        }

    };

    console.log(data);

}