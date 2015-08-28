function Analyse(button) {

    InputEnabled($(button), false);

    $.ajax({
        url: '?action=analysis',
        data: {
            compilation: $("#compilation").val(),
            compile_instructions: $("#compile-instructions").val(),
            output_directory: $("#analysis-directory").val(),
            chain: $("#chain-analysis-request").is(':checked'),
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

function Extract(button) {
    InputEnabled($(button), false)

    $.ajax({
        url: '?action=extract',
        data: {
            analysis_directory: $("#extract").val()
               },
        method: 'POST',
        success: function(data) {
            if (data.success) {
                Dialogue("Feature Extraction", "Extraction Enqueued", "", "/status");
            } else {
                Dialogue("Feature Extraction", "Extraction refused", data.reason ? data.reason : "Unknown reason", false, button);
            }
        },
        error: function(data) {
            Dialogue("Feature Extraction", "Error", "An error occurred processing request", false, button);
        }

    });
}