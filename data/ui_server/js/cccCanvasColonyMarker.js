function Blob(x, y, r, fill) {
    this.x = x || 0;
    this.y = y || 0;
    this.r = r || 1;
    this.fill = fill || '#AAAAAA';
}

Blob.prototype.draw = function(ctx) {
    ctx.fillStyle = this.fill;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, 2 * Math.PI, false);
    ctx.fill();
}

Blob.prototype.contains = function(mx, my) {
    var distancesquared = (mx - this.x) * (mx - this.x) + (my - this.y) * (my - this.y);
    return distancesquared <= this.r * this.r;
}

function CanvasState(canvas) {
    this.canvas = canvas;
    this.width = canvas.width;
    this.height = canvas.height;
    this.ctx = canvas.getContext('2d');
    var stylePaddingLeft, stylePaddingTop, styleBorderLeft, styleBorderTop;
    if (document.defaultView && document.defaultView.getComputedStyle) {
        this.stylePaddingLeft = parseInt(
            document.defaultView.getComputedStyle(
                canvas, null)['paddingLeft'],
            10) || 0;
        this.stylePaddingTop = parseInt(
            document.defaultView.getComputedStyle(
                canvas, null)['paddingTop'],
            10) || 0;
        this.styleBorderLeft = parseInt(
            document.defaultView.getComputedStyle(
                canvas, null)['borderLeftWidth'],
            10) || 0;
        this.styleBorderTop = parseInt(
            document.defaultView.getComputedStyle(
                canvas, null)['borderTopWidth'],
            10) || 0;
    }
    var html = document.body.parentNode;
    this.htmlTop = html.offsetTop;
    this.htmlLeft = html.offsetLeft;

    this.needsRender = true;
    this.shapes = [];
    this.dragging = false;
    this.selection = null;
    this.dragoffx = 0;
    this.dragoffy = 0;

    canvas.addEventListener(
        'selectstart',
        function(e) { e.preventDefault(); return false; }, false);

    canvas.addEventListener('mousedown', function (e) {
        var mouse = this.getMouse(e);
        var mx = mouse.x;
        var my = mouse.y;
        var shapes = this.shapes;
        var l = shapes.length;
        for (var i = l-1; i >= 0; i--) {
            if (shapes[i].contains(mx, my)) {
                var mySel = shapes[i];
                this.dragoffx = mx - mySel.x;
                this.dragoffy = my - mySel.y;
                this.dragging = true;
                this.selection = mySel;
                this.needsRender = true;
                return;
            }
        }
        // havent returned means we have failed to select anything.
        // If there was an object selected, we deselect it
        if (this.selection) {
            this.selection = null;
            this.needsRender = true; // Need to clear the old selection border
        }
    }.bind(this), true);
    canvas.addEventListener('mousemove', function(e) {
        if (this.dragging){
            var mouse = this.getMouse(e);
            this.selection.x = mouse.x - this.dragoffx;
            this.selection.y = mouse.y - this.dragoffy;
            this.needsRender = true; // Something's dragging so we must redraw
        }
    }.bind(this), true);
    canvas.addEventListener('mouseup', function(e) {
        this.dragging = false;
    }.bind(this), true);

    this.selectionColor = '#CC0000';
    this.selectionWidth = 2;
    this.interval = 30;
    setInterval(function() { this.draw(); }.bind(this), this.interval);
}

CanvasState.prototype.addShape = function(Blob) {
    this.shapes.push(Blob);
    this.needsRender = true;
}

CanvasState.prototype.clear = function() {
    this.ctx.clearRect(0, 0, this.width, this.height);
}

CanvasState.prototype.draw = function() {
    if (this.needsRender) {
        var ctx = this.ctx;
        var shapes = this.shapes;
        this.clear();

        var l = shapes.length;
        for (var i = 0; i < l; i++) {
            var Blob = shapes[i];
            if (Blob.x > this.width || Blob.y > this.height ||
                Blob.x + Blob.w < 0 || Blob.y + Blob.h < 0) continue;
            shapes[i].draw(ctx);
        }

        if (this.selection != null) {
            ctx.strokeStyle = this.selectionColor;
            ctx.lineWidth = this.selectionWidth;
            var mySel = this.selection;
            ctx.strokeRect(mySel.x,mySel.y,mySel.w,mySel.h);
        }

        this.needsRender = false;
    }
}

CanvasState.prototype.getMouse = function(e) {
    var element = this.canvas, offsetX = 0, offsetY = 0, mx, my;

    if (element.offsetParent !== undefined) {
        do {
            offsetX += element.offsetLeft;
            offsetY += element.offsetTop;
        } while ((element = element.offsetParent));
    }

    offsetX += this.stylePaddingLeft + this.styleBorderLeft + this.htmlLeft;
    offsetY += this.stylePaddingTop + this.styleBorderTop + this.htmlTop;

    mx = e.pageX - offsetX;
    my = e.pageY - offsetY;

    return {x: mx, y: my};
}
