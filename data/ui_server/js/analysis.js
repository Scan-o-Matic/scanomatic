function Analyse(button) {

    InputEnabled($(button), false);

    $.ajax({
        url: '?action=analysis',
        data: {
            compilation: $("#compilation").val(),
            compile_instructions: $("#compile-instructions").val(),
            output_directory: $("#analysis-directory").val()
               },
        method: 'POST',
        success: function(data) {
            if (data.success) {
                Dialogue("Analysis", "Analysis Enqueued", "", "/status");
            } else {
                Dialogue("Analysis", "Analysis Refused", data.reason ? data.reason : "Unknown reason", false, button);
            }
        },
        error: function(data) {
            Dialogue("Analysis", "Error", "An error occurred processing request", false, button);
        }

    });
}


function Extract() {

}