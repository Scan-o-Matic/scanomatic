
var localFixture = false;
var path = '';

function toggleLocalFixture(caller) {
    localFixture = $(caller).prop("checked");
    InputEnabled($(current_fixture_id), !localFixture);
}

function CompilePath(caller) {
    path = $(caller).val();
}

function Compile(button) {

    InputEnabled($(button), false);

    $.ajax({
        url: "?run=1",
        method: "POST",
        data: {'local': localFixture ? 1 : 0, 'fixture': $(current_fixture_id).val(),
               'path': path},
        success: function (data) {
            if (data.success) {
                $('<div class=\'dialog\'></div>').appendTo("body")
                            .prop("title", "Compile")
                            .html("<div><h3>Compilation has started.</h3></div>")
                            .dialog({modal: true,
                                     buttons: {
                                        Ok: function() {
                                            $(this).dialog("close");
                                            window.location ='/';
                                        }
                                     }
                            });
                $(button).closest("form")[0].reset();
            } else {
                $('<div class=\'dialog\'></div>').appendTo("body")
                        .prop("title", "Compile")
                        .html("<div><h3>Compilation refused</h3>Reason:\n<div class='indented'><em>" +
                            data.reason + "</em></div></div>")
                        .dialog({modal: true,
                                 buttons: {
                                    Ok: function() {
                                        $(this).dialog("close");
                                    }
                                 }
                        });
            }
            InputEnabled($(button), true);

        },
        error: function(data) {
            $('<div class=\'dialog\'></div>').appendTo("body")
                        .prop("title", "Compile")
                        .html("<div><h3>Unknown error occurred server side</h3></div>")
                        .dialog({modal: true,
                                 buttons: {
                                    Ok: function() {
                                        $(this).dialog("close");
                                    }
                                 }
                        });
            InputEnabled($(button), true);
        }});
}