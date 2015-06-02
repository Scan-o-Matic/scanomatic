
var grayscale_selector_class = ".grayscale-selector";

function get_grayscales(options) {
    console.log(options);
    options.empty();
    $.get("/grayscales?names=1", function(data, status) {
        if (data.grayscales) {
            for (var i=0; i<data.grayscales.length; i++)
                options.append($("<option />").val(data.grayscales[i]).text(data.grayscales[i]));
        }
    });

}

function LoadGrayscales(){
    Execute(grayscale_selector_class, get_grayscales);
}

function GetSelectedGrayscale(identifier) {
    vals = []
    $(grayscale_selector_class).each( function(i, obj) {
        obj = $(obj);
        if (identifier == null|| obj.id)
            vals.push(obj.val());
    });

    return vals[0];
}