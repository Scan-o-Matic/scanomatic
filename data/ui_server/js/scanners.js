function get_free_scanners(target_id) {
    var target = $(target_id);
    target.empty();
    $.get("/scanners/free", function(data, status) {

        if (data.success) {
            $.each(data.scanners, function(key, value) {
                target.append($("<option />").val(key).text(value));
            });
        }

        unselect(target);
    });
}