let pmType;

function toggleVisibilities(button, state) {
    $(button).parent().nextAll().css('display', state ? '' : 'none');
    $('#scanner_section').css('display', $(button).val() !== 'notInstalled' ? '' : 'none');
    pmType = $(button).val();
}

function dynamicallyLimitScanners(button) {
    const numOfScanners = $('#number_of_scanners');
    const maxVal = $(button).val();
    numOfScanners.attr('max', maxVal);
    if (numOfScanners.val() > maxVal) {
        numOfScanners.val(maxVal);
    }
}

function UpdateSettings(button) {

    InputEnabled($(button), false);

    const data = {
        number_of_scanners: $('#number_of_scanners').val(),
        power_manager: {
            sockets: $('#number_of_sockets').val(),
            type: pmType,
            host: $('#power_manager_host').val(),
            mac: $('#power_manager_mac').val(),
            name: $('#power_manager_name').val(),
            password: $('#power_manager_password').val(),
        },
        paths: {
            projects_root: $('#projects_root').val(),
        },
        computer_human_name: $('#computer_human_name').val(),
        mail: {
            warn_scanning_done_minutes_before: $('#warn_scanning_done_minutes_before').val(),
        },
    };

    API.postJSON('/api/settings', data)
        .then(() => Dialogue('Settings', 'Settings updated', '', '/settings'))
        .catch((reason) => {
            if (reason) {
                Dialogue('Settings', 'Settings refused', reason, false, button);
            } else {
                Dialogue('Settings', 'Error', 'An error occurred processing request', false, button);
            }
        });
}
