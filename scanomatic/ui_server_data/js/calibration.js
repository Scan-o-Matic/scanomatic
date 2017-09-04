
function createSelector(elem) {

    $(elem).empty();
    $(elem).prop('disabled', true);
    $('#active-ccc-error').remove();

    $.getJSON('/api/calibration/active')
        .done(function(data) {
            $.each(data.cccs, function(i, item) {
                $(elem).append($('<option>', {
                    value: item.id ? item.id : '',
                    text: item.species + ', ' + item.reference
                }));
            });
            $(elem).prop('disabled', false);
        })
        .fail(function(data) {
            $(elem).after('<div class=\'error\' id=\'active-ccc-error\'>Could not retrieve CCCs</div>');
        });
}
