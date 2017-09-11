//TODO: Consider using https://github.com/image-js/tiff-js if needed

function browserSupportsImages() {
    // Check for the various File API support.
    if (window.File && window.FileReader && window.FileList && window.Blob) {
        return true;
    } else {
        return false;
    }
}

function handleFileSelect(evt, callback) {
    evt.stopPropagation();
    evt.preventDefault();
    var files;

    if (evt.target && evt.target.files) {
        files = evt.target.files;
    } else if (evt.dataTransfer && evt.dataTransfer.files) {
        files = evt.dataTransfer.files;
    } else {
        console.warn("Event not supported" + evt);
        return;
    }

    for (var i = 0, f; f = files[i]; i++) {

          // Only process image files.
      if (!f.type.match('image.*')) {
        continue;
      }

      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = (function(arrayBuffer) {
        return function(e) {
            callback(arrayBuffer);
        };
      })(f);

      // Read in the image file as a data URL.
      reader.readAsArrayBuffer(f);
    }
}

function registerFileSelector(elemId, callback)
  document.getElementById(elemId).addEventListener('change', function(evt) {handleFileSelect(evt, callback);}, false);

function registerDropZone(elemId, callback) {
  var dropZone = document.getElementById(elemId);
  dropZone.addEventListener('dragover', handleDragOver, false);
  dropZone.addEventListener('drop', function(evt) {handleFileSelect(evt, callback);}, false);
}