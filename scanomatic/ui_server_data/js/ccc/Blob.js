export default function Blob(x, y, r, fill) {
    this.x = x || 0;
    this.y = y || 0;
    this.r = r || 1;
    this.fill = fill || '#AAAAAA';
}

Blob.prototype.draw = function draw(ctx) {
    ctx.fillStyle = this.fill;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, 2 * Math.PI, false);
    ctx.fill();
};

Blob.prototype.contains = function contains(mx, my) {
    const distancesquared = (mx - this.x) * (mx - this.x) + (my - this.y) * (my - this.y);
    return distancesquared <= this.r * this.r;
};
