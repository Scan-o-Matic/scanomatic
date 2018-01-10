function UpdateSettings(button) {

    InputEnabled($(button), false);

    const data = {
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
