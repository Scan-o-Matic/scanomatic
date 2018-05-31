if (!d3.scanomatic) d3.scanomatic = {};

function DrawCurves(container, time, serRaw, serSmooth, gt, gtWhen, yld) {
    //GrowthChart
    var chartMarginAll = 30;
    var chartMargin = { top: chartMarginAll, right: chartMarginAll, bottom: chartMarginAll, left: chartMarginAll };
    var chartwidth = 350;
    var chartheight = 294;
    d3.select(container).select('svg').remove();
    var chart = d3.select(container)
        .append("svg")
        .attr({
            "width": chartwidth,
            "height": chartheight,
            "class": "growthChart",
            "margin": 3
        });

    var defs = chart.append("defs");

    defs.append("marker")
            .attr({
                "id": "arrow",
                "viewBox": "0 -5 10 10",
                "refX": 5,
                "refY": 0,
                "markerWidth": 4,
                "markerHeight": 4,
                "orient": "auto"
            })
            .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("class", "arrowHead");

    defs.append("marker")
            .attr({
                "id": "arrow2",
                "viewBox": "0 -5 10 10",
                "refX": 5,
                "refY": 0,
                "markerWidth": 4,
                "markerHeight": 4,
                "orient": "auto-start-reverse"
            })
            .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("class", "arrowHead");

    // chart
    var gChart = d3.scanomatic.growthChart(time, serRaw, serSmooth);
    gChart.width(chartwidth);
    gChart.height(chartheight);
    gChart.margin(chartMargin);
    gChart.generationTimeWhen(gtWhen);
    gChart.generationTime(gt);
    gChart.growthYield(yld);
    gChart(chart);
}

d3.scanomatic.growthChart = function (time, serRaw, serSmooth) {

    //properties
    var margin;
    var height;
    var width;
    var generationTimeWhen;
    var generationTime;
    var growthYield;
    let drawn = false;

    //local variables
    var g;


    function chart(container) {
        g = container.append("g")
        .attr({
            "transform": "translate(" + margin.left + "," + margin.top + ")",
            "class": "PlotArea"
        });
        update();
    }

    chart.update = update;

    function update() {
        //data
        if (!serSmooth || !serRaw || !time) {
            drawn = false;
        } else {
            if (drawn) return;
        };
        //chart
        var w = width - margin.left - margin.right;
        var h = height - margin.top - margin.bottom;

        if (serRaw.length !== time.length || serSmooth.length !== time.length)
            throw ("GrowthData lenghts do not match!!!");
        drawn = true;

        var odExtend = getExtentFromMultipleArrs(serRaw, serSmooth);

        var rawData = getDataObject(time, serRaw);
        var smoothData = getDataObject(time, serSmooth);

        var xScale = d3.scale.linear()
            .domain(d3.extent(time))
            .range([0, w]);

        var yScale = d3.scale.log()
            .base(2)
            .domain(d3.extent(odExtend))
            .range([h, 0]);

        addAxis(xScale, yScale, h);

        addSeries(rawData, smoothData, xScale, yScale);

        addMetaGt(smoothData, xScale, yScale);

        addMetaYield(smoothData, xScale, yScale);

        function getDataObject(time, value) {
            var dataObject = [];
            var i = 0;
            time.forEach(function (timePoint) {
                var p = { time: timePoint, value: value[i] }
                dataObject.push(p);
                i++;
            });
            return dataObject;
        }
    }

    function addAxis(xScale, yScale, chartHeight) {

        var xAxis = d3.svg.axis()
           .scale(xScale)
           .orient("bottom");

        var yAxis = d3.svg.axis()
            .scale(yScale)
            .orient("left")
            //.ticks(5)
            .tickFormat(d3.format(".0e"));

        g.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + chartHeight + ")")
            .call(xAxis);

        g.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", ".5em")
            .style("text-anchor", "end")
            .style("font-size","11")
            .text("Population size [cells]");
    }

    function addSeries(rawData, smoothData, xScale, yScale) {

        var lineFun = d3.svg.line()
            .x(function (d) { return xScale(d.time); })
            .y(function (d) { return yScale(d.value); })
            .interpolate("linear");

        g.append("g")
           .classed("series", true)
           .append("path")
           .attr({
               "class": "raw",
               d: lineFun(rawData)
           });

        g.append("g")
            .classed("series", true)
            .append("path")
            .attr({
                "class": "smooth",
                d: lineFun(smoothData)
            });
    }

    function addMetaGt(smoothData, xScale, yScale) {

        if (generationTime == null || generationTimeWhen == null) return;

        var lineStartOffset = 50;
        var lineEndOffset = 10;

        var smoothGtTime = generationTimeWhen;
        var smoothGtValue = 0;
        for (var i = 0; i < smoothData.length; i++) {
            var approx = d3.round(smoothData[i].time, 2);
            if (approx == d3.round(smoothGtTime, 2))
                smoothGtValue = smoothData[i].value;
        }

        var gtX = xScale(smoothGtTime);
        var gtY = yScale(smoothGtValue);

        var gMetaGt = g.append("g")
          .classed("meta gt", true);

        gMetaGt.append("circle")
            .attr({
                "cx": gtX,
                "cy": gtY,
                "r": 4,
                "fill": "red"
            });

        gMetaGt.append("line")
            .attr({
                "class": "arrow",
                "marker-end": "url(#arrow)",
                "x1": gtX,
                "y1": getYOffset(gtY, lineStartOffset),
                "x2": gtX,
                "y2": getYOffset(gtY, lineEndOffset),
                "stroke": "black",
                "stroke-width": 1.5
            });

        gMetaGt.append("text")
            .attr({
                x: gtX - 10,
                y: getYOffset(gtY, 55)
            })
            .text("GT");
        //Py=m (Px-x) + Py
        console.log("GT:" + generationTime);
        console.log("GTTimeWhen:" + generationTimeWhen);
        console.log("GTTimeWhenValue:" + smoothGtValue);
        var windowSize = 4;
        var gtSlope = 1 / generationTime;
        var l = parseFloat(smoothGtTime) - windowSize;
        var leftXLimit = xScale(l);
        var logPy = getBaseLog(2, smoothGtValue);
        var yLeftLogged = logPy - (windowSize * gtSlope);
        var yLeft = Math.pow(2, yLeftLogged);
        var leftYLimit = yScale(yLeft);

        gMetaGt.append("line")
            .attr({
                "x1": gtX,
                "y1": gtY,
                "x2": leftXLimit,
                "y2": leftYLimit,
                "stroke": "blue",
                "stroke-width": 3
            });

        var r = parseFloat(smoothGtTime) + windowSize;
        var rightXlimit = xScale(r);
        var yRightLogged = (windowSize * gtSlope) + logPy;
        var yRight = Math.pow(2, yRightLogged);
        var rightYLimit = yScale(yRight);

        gMetaGt.append("line")
            .attr({
                "x1": gtX,
                "y1": gtY,
                "x2": rightXlimit,
                "y2": rightYLimit,
                "stroke": "blue",
                "stroke-width": 3
            });
    }

    function addMetaYield(smoothData, xScale, yScale) {

        if (growthYield == null) return;

        var smoothYieldValue = growthYield;
        var smoothYieldTime = 0;
        for (var i = 0; i < smoothData.length; i++) {
            if (smoothData[i].value >= smoothYieldValue) {
                smoothYieldTime = smoothData[i].time;
                break;
            }
        }

        var gtX = xScale(smoothYieldTime);
        var gtY = yScale(smoothYieldValue);
        var baseX = gtX;
        var baseY = yScale(smoothData[0].value);
        console.log("Yield time:" + smoothYieldTime);
        console.log("Yield value:" + smoothYieldValue);

        var gMetaYeild = g.append("g")
          .classed("meta yield", true);

        gMetaYeild.append("line")
            .attr({
                "class": "arrow",
                "marker-end": "url(#arrow)",
                "marker-start": "url(#arrow2)",
                "x1": gtX,
                "y1": gtY + 5,
                "x2": baseX,
                "y2": baseY - 5,
                "stroke": "black",
                "stroke-width": 1.5
            });

        var middle =(baseY - gtY) / 2;
        gMetaYeild.append("text")
            .attr({
                x: gtX +3,
                y: gtY + middle
            })
            .text("Yield");
    }

    function getYOffset(y, offset) {
        return (y > 100) ? y - offset : y + offset;
    }

    chart.margin = function (value) {
        if (!arguments.length) return margin;
        margin = value;
        return chart;
    }

    chart.width = function (value) {
        if (!arguments.length) return width;
        width = value;
        return chart;
    }

    chart.height = function (value) {
        if (!arguments.length) return height;
        height = value;
        return chart;
    }

    chart.generationTimeWhen = function (value) {
        if (!arguments.length) return generationTimeWhen;
        generationTimeWhen = value;
        return chart;
    }

    chart.generationTime = function (value) {
        if (!arguments.length) return generationTime;
        generationTime = value;
        return chart;
    }

    chart.growthYield = function (value) {
        if (!arguments.length) return growthYield;
        growthYield = value;
        return chart;
    }

    return chart;

}
