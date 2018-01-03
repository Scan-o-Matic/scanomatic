function get_free_scanners(target_id) {
    const target = $(target_id);
    target.empty();
    API.get('/api/status/scanners/free')
        .then((data) => {
            $.each(data.scanners, (key, value) => {
                target.append($('<option />').val(key).text(value));
            });
            unselect(target);
        });
}
