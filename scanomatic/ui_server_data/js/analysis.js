var gridplates = null;
var localFixture = true;
var path = '';

function analysisToggleLocalFixture(caller) {
    localFixture = $(caller).prop('checked');
    InputEnabled($(current_fixture_id), !localFixture);
    set_fixture_plate_listing();
}

function set_fixture_plate_listing() {
    callback = function (data, status) {
        if (!data.success) {
            $('#fixture-error-message').html(`<em>${data.reason}</em>`).show();
        } else {
            $('#fixture-error-message').hide();
            gridplates = ArrMap(data.plates, e => e.index);
            if ($('#manual-regridding').prop('checked')) {
                $('#manual-regridding-settings').show();
            } else {
                $('#manual-regridding-settings').hide();
            }
            parent = $('#manual-regridding-plates');
            parent.empty();
            ArrMap(gridplates, (e) => { append_regridding_ui(parent, e); });
        }
    };

    error_callback = function () {
        $('#fixture-error-message').html('<em>Fixture file missing</em>').show();
    };

    if (localFixture) {
        if (path.length > 5) {
            $.get(`/api/data/fixture/local/${path.substring(5, path.length)}`, callback).fail(error_callback);
        } else {
            error_callback();
        }
    } else {
        fixt = $(current_fixture_id).val();
        if (fixt) {
            $.get(`/api/data/fixture/get/${fixt}`, callback).fail(error_callback);
        } else {
            $('#fixture-error-message').hide();
        }
    }
}

function append_regridding_ui(parent, plate_index) {
    parent.append(
        "<div class='plate-regridding' id='plate-regridding-" + plate_index + "' onmouseleave='hidegridimage();'>" +
            '<fieldset>' +
            "<img class='grid_icon' src='/images/grid_icon.png' onmouseenter='loadgridimage(" + (plate_index - 1) + ");'>" +
            '<legend>Plate ' +  plate_index + '</legend>' +

            "<input type='radio' name='plate-regridding-radio-" + plate_index + "' value='Keep' checked='checked'>" +
            "<label id='plate-regridding-keep" + plate_index + "'>Keep previous</label><br>" +

            "<input type='radio' name='plate-regridding-radio-" + plate_index + "' value='Offset'>" +
            "<label id='plate-regridding-offset" + plate_index + "'>Offset</label>" +
            "<input type='number' class='plate-offset' id='plate-regridding-offset-d1-" + plate_index + "' value='0' name='Offset-d1'>" +
            "<input type='number' class='plate-offset' id='plate-regridding-offset-d2-" + plate_index + "' value='0' name='Offset-d2'><br>" +

            "<input type='radio' name='plate-regridding-radio-" + plate_index + "' value='New'>" +
            "<label id='plate-regridding-new" + plate_index + "'>New grid from scratch</label><br>" +
            '</fieldset>' +
        '</div>'
    );
}

show_gridimage = false;

function hidegridimage() {

    show_gridimage = false;
    $('#manual-regridding-image').hide();

}

function loadgridimage(i) {

    show_gridimage = true;
    curDir = get_dir();
    $('#manual-regridding-image').empty();
    $.get(
        '/api/results/gridding/' + i + curDir.substring(4, curDir.length) + '/' + $('#manual-regridding-source-folder').val(),
        function (data) {
            $('#manual-regridding-image').append(data.documentElement);
        }
    ).fail(function() {
        $('#manual-regridding-image').append("<p class='error-message'>Could not find the grid image! Maybe gridding failed last time?</p>");
    }).always(function() {
        if (show_gridimage) {
            $('#manual-regridding-image').show();
        } else {
            $('#manual-regridding-image').hide();
        }
    });
}

function regridding_settings_data() {
    max = Math.max.apply(Math, gridplates);
    plates = [];
    for (i=1;i<=max; i++) {
        plates.push(get_regridding_setting(i));
    }
    return plates;
}

function get_regridding_setting(i) {

    e = $('#plate-regridding-' + i);
    if (e.length != 0) {
        switch (e.find('input[name=plate-regridding-radio-' + i + ']:checked').val()) {
            case 'Keep':
                return [0, 0];
            case 'Offset':
                return [
                    parseInt(e.find('#plate-regridding-offset-d1-' + i).val()),
                    parseInt(e.find('#plate-regridding-offset-d2-' + i).val()),
                ];

            case 'New':
                return null;
            default:
                return null;
        };
    } else {
        return null;
    }
}

function get_dir() {
    return $('#compilation').val().replace(/\/[^\/]*$/,'');
}

function set_regridding_source_directory(input) {
    path = get_dir();
    get_path_suggestions(
        input,
        true,
        '',
        null,
        function(data, status) {

            //TODO: For some reason popup don't appear...

            regrid_chkbox = $('#manual-regridding');
            regrid_chkbox
                .prop('disabled', data.has_analysis ? false : true)
                .prop('checked', data.has_analysis ? true : false);

            toggleManualRegridding(regrid_chkbox);

        },
        path,
        true);
}

function toggleManualRegridding(chkbox) {
    const isActive = $(chkbox).prop('checked');
    if (isActive) {
        $('#manual-regridding-settings').show();
    } else {
        $('#manual-regridding-settings').hide();
    }
}

function set_analysis_directory(input, validate) {

    get_path_suggestions(
        input,
        true,
        '',
        null,
        function(data, status) {
            if (validate) {
                InputEnabled($('#submit-button2'), data.valid_parent && data.exists);
            }

    });
}

function set_file_path(input, suffix, suffix_pattern, toggle_regridding_if_not_exists) {

    get_path_suggestions(
        input,
        false,
        suffix,
        suffix_pattern,
        function(data, status) {

            if (toggle_regridding_if_not_exists) {
                $('#manual-regridding-source-folder').prop('disabled', !data.exists);
            }

            if (localFixture) {
                set_fixture_plate_listing();
            }
        }
    );
}

function Analyse(button) {

    InputEnabled($(button), false);

    const data = {
        compilation: $('#compilation').val(),
        compile_instructions: $('#compile-instructions').val(),
        output_directory: $('#analysis-directory').val(),
        ccc: $('#ccc-selection').val(),
        chain: $('#chain-analysis-request').is(':checked') ? 0 : 1,
        one_time_positioning: $('#one_time_positioning').is(':checked') ? 0 : 1,
    };

    if ($('#manual-regridding').prop('checked')) {
        data['reference_grid_folder'] = $('#manual-regridding-source-folder').val();
        data['gridding_offsets'] = regridding_settings_data();
    }

    API.postJSON('/api/project/analysis', data)
        .then(() => Dialogue('Analysis', 'Analysis Enqueued', '', '/status'))
        .catch((reason) => {
            if (reason) {
                Dialogue('Analysis', 'Analysis Refused', reason, false, button);
            } else {
                Dialogue('Analysis', 'Error', 'An error occurred processing request', false, button);
            }
        });
}

function Extract(button) {
    InputEnabled($(button), false)

    API.postJSON(
        '/api/project/feature_extract',
        {
            analysis_directory: $('#extract').val(),
            keep_qc: $('#keep-qc').is(':checked') ? 0 : 1,
        },
    )
        .then(() => Dialogue('Feature Extraction', 'Extraction Enqueued', '', '/status'))
        .catch((reason) => {
            if (reason) {
                Dialogue('Feature Extraction', 'Extraction refused', reason, false, button);
            } else {
                Dialogue('Feature Extraction', 'Unexpected error', 'An error occurred processing request', false, button);
            }
        });
}

function BioscreenExtract(button) {
    InputEnabled($(button), false)

    API.postJSON(
        '/api/project/feature_extract/bioscreen',
        {
            bioscreen_file: $('#bioscreen_extract').val(),
        },
    )
        .then(() => Dialogue('Feature Extraction', 'Extraction Enqueued', '', '/status'))
        .catch((reason) => {
            if (reason) {
                Dialogue('Feature Extraction', 'Extraction refused', reason, false, button);
            } else {
                Dialogue('Feature Extraction', 'Unexpected error', 'An error occurred processing request', false, button);
            }
        });
}
