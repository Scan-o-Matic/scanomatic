var classExperimentSelected = "ExperimentSelected";
var dispatcherSelectedExperiment = "SelectedExperiment";
var markingSelection = false;
var allowMarking = false;
var plateMetaDataType = {
    OK: "OK",
    BadData: "BadData",
    Empty: "Empty",
    NoGrowth: "NoGrowth",
    UndecidedProblem: "UndecidedProblem"
}

if (!d3.scanomatic) d3.scanomatic = {};


function executeFunctionByName(functionName, context /*, args */) {
    var args = Array.prototype.slice.call(arguments, 2);
    var namespaces = functionName.split(".");
    var func = namespaces.pop();
    for (var i = 0; i < namespaces.length; i++) {
        context = context[namespaces[i]];
    }
    return context[func].apply(context, args);
}

function DrawPlate(container, data, growthMetaData, plateMetaData, phenotypeName, dispatch) {

    //plate
    var cols = data[0].length;
    var rows = data.length;
    //experiment
    var circleRadius = 4;
    var circleMargin = 1;
    var cellSize = (circleRadius * 2) + circleMargin;
    //heatmap
    var heatmapMargin = 7;
    var scaleWidth = 30;
    var gridwidth = (cols * cellSize) + scaleWidth + (heatmapMargin * 2);
    var gridheight = (rows * cellSize) + (heatmapMargin * 2);
    var colorScheme = ["blue", "white", "red"];
    //heatmap legend
    var legendWidth = 25;
    var legendMargin = 5;

    //SetDebugText(data, cols, rows);

    var grid = d3.select(container)
        .append("svg")
        .attr({
            "width": gridwidth + legendMargin + legendWidth,
            "height": gridheight,
            "class": "PlateHeatMap"
        });


    addSelectionHanddling(grid);
    addSymbolsToSGV(grid);

    var plateGroup = grid.append("g").classed("gHeatmap", true);

    //heatmap
    var heatMap = d3.scanomatic.plateHeatmap();
    heatMap.data(data);
    heatMap.phenotypeName(phenotypeName);
    heatMap.growthMetaData(growthMetaData);
    heatMap.plateMetaData(plateMetaData);
    heatMap.cellSize(cellSize);
    heatMap.cellRadius(circleRadius);
    heatMap.setColorScale(colorScheme);
    heatMap.margin(heatmapMargin);
    heatMap.legendWidth(legendWidth);
    heatMap.legendMargin(legendMargin);
    heatMap.displayLegend(true);
    heatMap.dispatch2(dispatch);
    heatMap(plateGroup);
    //heatMap.on(dispatcherSelectedExperiment, function (datah) {
    //    console.log("dispatched:" + datah);
    //});
    return d3.rebind(DrawPlate, heatMap, "on");
};

function addSelectionHanddling(svgRoot) {
    svgRoot.on("mousedown",
            function () {
                if (!allowMarking) return;

                if (!d3.event.ctrlKey) {
                    d3.selectAll("g.expNode.selected").classed("selected", false);
                }

                var p = d3.mouse(this);

                svgRoot.append("rect")
                    .attr({
                        "rx": 6,
                        "ry": 6,
                        "class": "selection",
                        "x": p[0],
                        "y": p[1],
                        "width": 0,
                        "height": 0
                    });

            })
        .on("mousemove",
            function () {
                if (!allowMarking) return;
                var s = svgRoot.select("rect.selection");
                if (!s.empty()) {
                    markingSelection = true;
                    var p = d3.mouse(this);
                    var d = {
                        x: parseInt(s.attr("x"), 10),
                        y: parseInt(s.attr("y"), 10),
                        width: parseInt(s.attr("width"), 10),
                        height: parseInt(s.attr("height"), 10)
                    };
                    var move = {
                        x: p[0] - d.x,
                        y: p[1] - d.y
                    };
                    if (move.x < 1 || (move.x * 2 < d.width)) {
                        d.x = p[0];
                        d.width -= move.x;
                    } else { d.width = move.x; }
                    if (move.y < 1 || (move.y * 2 < d.height)) {
                        d.y = p[1];
                        d.height -= move.y;
                    } else { d.height = move.y; }
                    s.attr(d);

                    d3.selectAll("g.expNode.selection.selected").classed("selected", false);

                    d3.selectAll(".plateWell")
                        .each(function (stateData) {
                            if (
                                !d3.select(this).classed("selected") &&
                                    stateData.celStartX >= d.x &&
                                    stateData.celEndX <= d.x + d.width &&
                                    stateData.celStartY >= d.y &&
                                    stateData.celEndY <= d.y + d.height
                            ) {

                                d3.select(this.parentNode)
                                    .classed("selection", true)
                                    .classed("selected", true);
                            }
                        });
                }
            })
        .on("mouseup",
            function () {
                if (!allowMarking) return;
                markingSelection = false;
                svgRoot.selectAll("rect.selection").remove();
                d3.selectAll('g.expNode.selection').classed("selection", false);
            })
        .on("mouseout",
            function () {
                if (d3.event.relatedTarget != null && d3.event.relatedTarget.tagName == 'HTML') {
                    svgRoot.selectAll("rect.selection").remove();
                    d3.selectAll('g.expNode.selection').classed("selection", false);
                }
            });
}

function getValidSymbol(type) {
    switch (type) {
        case plateMetaDataType.BadData:
            return "symBadData";
        case plateMetaDataType.Empty:
            return "symEmpty";
        case plateMetaDataType.OK:
            return "symOK";
        case plateMetaDataType.NoGrowth:
            return "symNoGrowth";
        case plateMetaDataType.UndecidedProblem:
            return "symUndecided2";

        default:
            return null;
    }
}

function addSymbolsToSGV(svgRoot) {
    addSymbolToSGV(svgRoot, plateMetaDataType.BadData);
    addSymbolToSGV(svgRoot, plateMetaDataType.Empty);
    addSymbolToSGV(svgRoot, plateMetaDataType.NoGrowth);
    addSymbolToSGV(svgRoot, plateMetaDataType.OK);
    addSymbolToSGV(svgRoot, plateMetaDataType.UndecidedProblem);
}

function addSymbolToSGV(svgRoot, type) {

    switch (type) {
        case plateMetaDataType.BadData:
            var badDataSym = svgRoot.append("symbol")
            .attr({
                "id": "symBadData",
                "viewBox": "0 0 100 125"
            });
                badDataSym.append("path")
                    .attr("d", "M94.202,80.799L55.171,13.226c-1.067-1.843-3.022-2.968-5.147-2.968c-0.008,0-0.016,0-0.023,0s-0.016,0-0.023,0  c-2.124,0-4.079,1.125-5.147,2.968L5.798,80.799c-1.063,1.85-1.063,4.124,0,5.969c1.057,1.846,3.024,2.976,5.171,2.976h78.063  c2.146,0,4.114-1.13,5.171-2.976C95.266,84.923,95.266,82.646,94.202,80.799z M14.412,81.79L50,20.182L85.588,81.79H14.412z");
                badDataSym.append("polygon")
                    .attr("points", "64.512,70.413 56.414,62.164 64.305,54.188 57.873,47.826 50.075,55.709 42.212,47.7 35.757,54.038 43.713,62.141 35.489,70.455 41.92,76.817 50.051,68.598 58.057,76.751");

                svgRoot.append("symbol")
                    .attr({
                        "id": "symNoGrowth",
                        "viewBox": "0 0 100 125"
                    })
                    .append("path")
                        .attr("d", "M50,95c24.853,0,45-20.147,45-45C95,25.147,74.853,5,50,5S5,25.147,5,50C5,74.853,25.147,95,50,95z M25,45h50v10H25V45z");
            //CREDIT: Created by Arthur Shlain from from the Noun Project
            break;
        case plateMetaDataType.Empty:
            svgRoot.append("symbol")
            .attr({
                "id": "symEmpty",
                "viewBox": "0 0 24 30"
            })
            .append("path")
                .attr("d", "M22.707,1.293c-0.391-0.391-1.023-0.391-1.414,0L16.9,5.686C15.546,4.633,13.849,4,12,4c-4.418,0-8,3.582-8,8  c0,1.849,0.633,3.546,1.686,4.9l-4.393,4.393c-0.391,0.391-0.391,1.023,0,1.414C1.488,22.902,1.744,23,2,23s0.512-0.098,0.707-0.293  L7.1,18.314C8.455,19.367,10.151,20,12,20c4.418,0,8-3.582,8-8c0-1.849-0.633-3.545-1.686-4.9l4.393-4.393  C23.098,2.316,23.098,1.684,22.707,1.293z M6,12c0-3.309,2.691-6,6-6c1.294,0,2.49,0.416,3.471,1.115l-8.356,8.356  C6.416,14.49,6,13.294,6,12z M18,12c0,3.309-2.691,6-6,6c-1.294,0-2.49-0.416-3.471-1.115l8.356-8.356C17.584,9.51,18,10.706,18,12z");
                //CREDIT: Created by ? from from the Noun Project
            break;
        case plateMetaDataType.OK:
            var okSym = svgRoot.append("symbol")
           .attr({
               "id": "symOK",
               "viewBox": "0 0 90 112.5"
           });
                okSym.append("path")
                    .attr("d", "M38.3,61.3L38.3,61.3c-0.9,0-1.7-0.3-2.3-1L25.9,50.2c-1.3-1.3-1.3-3.4,0-4.7c1.3-1.3,3.4-1.3,4.7,0l7.8,7.8l22.6-22.6   c1.3-1.3,3.4-1.3,4.7,0c1.3,1.3,1.3,3.4,0,4.7L40.7,60.3C40.1,60.9,39.2,61.3,38.3,61.3z");
                okSym.append("path")
                    .attr("d", "M45.7,81.3C26,81.3,9.9,65.3,9.9,45.5C9.9,25.8,26,9.7,45.7,9.7s35.8,16.1,35.8,35.8C81.6,65.3,65.5,81.3,45.7,81.3z    M45.7,16.3c-16.1,0-29.2,13.1-29.2,29.2s13.1,29.2,29.2,29.2S75,61.6,75,45.5S61.9,16.3,45.7,16.3z");
            //CREDIT: Created by Louis Buck from from the Noun Project
            break;
        case plateMetaDataType.NoGrowth:
            break;
        case plateMetaDataType.UndecidedProblem:
            //svgRoot.append("symbol")
            //.attr({
            //    "id": "symUndecided",
            //    "viewBox": "0 0 100 100"
            //})
            //.append("path")
            //    .attr("d", "M50.1,5c-13,0-23.7,10.7-23.7,23.7c0,3.9,3.3,7.2,7.3,7.2c4.1,0,7.3-3.3,7.3-7.2c0-5,4.1-9.1,9.1-9.1c4.9,0,8.9,4.1,8.9,9.1  c0,4.6-2,6.9-5.7,11.1c-4.6,4.7-10.7,11.2-10.7,23.6c0,4.2,3.3,7.4,7.4,7.4c4,0,7.2-3.2,7.2-7.4c0-6.7,2.8-9.7,6.6-13.7  c4.3-4.7,9.6-10.4,9.6-21.1C73.6,15.7,63.1,5,50.1,5L50.1,5z M50.1,77.4c-4.9,0-8.9,3.8-8.9,8.8c0,4.9,4,8.8,8.9,8.8  c4.7,0,8.8-3.9,8.8-8.8C58.9,81.3,54.9,77.4,50.1,77.4z");
            ////CREDIT: Created by Ervin Bolat from from the Noun Project
            //undecided 2
            svgRoot.append("symbol")
            .attr({
                "id": "symUndecided2",
                "viewBox": "0 0 100 125"
            })
            .append("path")
            .attr("d", "M43.475,69.043c0,1.633,0.518,2.965,1.551,4.02c1.057,1.033,2.369,1.572,3.941,1.572   c1.594,0,2.926-0.539,3.979-1.572c1.074-1.055,1.613-2.387,1.613-4.02c0-1.613-0.539-2.965-1.613-3.998   c-1.053-1.057-2.385-1.574-3.979-1.574c-1.572,0-2.885,0.518-3.941,1.574C43.992,66.078,43.475,67.43,43.475,69.043z    M45.363,57.959h7.504c-0.201-1.531-0.1-2.289,0.359-3.521c0.475-1.234,1.092-2.408,1.869-3.5c0.756-1.098,1.631-2.15,2.607-3.166   c0.953-0.996,1.869-2.051,2.705-3.143c0.836-1.096,1.551-2.25,2.109-3.463c0.576-1.236,0.855-2.588,0.855-4.119   c0-1.971-0.338-3.68-0.994-5.154c-0.678-1.492-1.633-2.727-2.865-3.721c-1.236-1.016-2.709-1.77-4.379-2.268   c-1.691-0.498-3.521-0.758-5.512-0.758c-2.727,0-5.172,0.598-7.342,1.771c-2.189,1.154-4.08,2.646-5.652,4.479l4.756,4.217   c1.074-1.154,2.25-2.049,3.48-2.725c1.256-0.658,2.629-0.996,4.16-0.996c1.893,0,3.365,0.518,4.438,1.57   c1.055,1.057,1.594,2.43,1.594,4.16c0,1.096-0.279,2.109-0.816,3.064c-0.559,0.957-1.236,1.93-2.051,2.906   c-0.816,0.973-1.691,1.99-2.605,3.004c-0.918,1.035-1.732,2.127-2.449,3.322c-0.715,1.195-1.254,2.51-1.631,3.939   C45.127,55.293,45.107,56.25,45.363,57.959z M11,50c0,21.531,17.471,39,39,39c21.531,0,39-17.469,39-39S71.531,11,50,11   C28.471,11,11,28.469,11,50z M18.184,50c0-17.57,14.246-31.818,31.816-31.818S81.818,32.43,81.818,50S67.57,81.818,50,81.818   S18.184,67.57,18.184,50z");
            //CREDIT: Created by Ervin Bolat from from the Noun Project
            break;

        default:
            return null;
    }
}

function SetDebugText(data, cols, rows) {

    d3.select("#text").append("p").text("cols = " + cols);
    d3.select("#text").append("p").text("rows = " + rows);

    var min = d3.min(data[0]);
    var max = d3.max(data[0]);
    var mean = d3.mean(data[0]);
    d3.select("#text").append("p").text("min = " + min);
    d3.select("#text").append("p").text("max = " + max);
    d3.select("#text").append("p").text("mean = " + mean);
}

d3.scanomatic.plateHeatmap = function () {

    //properties
    var data;
    var growthMetaData;
    var plateMetaData;
    var cellSize;
    var cellRadius;
    var colorScale;
    var colorSchema;
    var margin;
    var displayLegend;
    var legendMargin;
    var legendWidth;
    var phenotypeName;
    var dispatch2;

    // local variables
    var g;
    var cols;
    var rows;
    var phenotypeMin;
    var phenotypeMax;
    var phenotypeMean;
    var heatMapCelWidth;
    var heatMapCelHeight;
    var dispatch = d3.dispatch(dispatcherSelectedExperiment);
    var numberFormat = "0.3n";

    function heatmap(container) {
        g = container;
        update();
    }

    heatmap.update = update;
    function update() {
        dispatch2.on("setExp", setExperiment);
        dispatch2.on("reDrawExp", reDrawExperiment);
        var plateData = composeMetadata;
        var toolTipDiv = d3.select("#toolTipDiv");
        if (toolTipDiv.empty()) {
            toolTipDiv = d3.select("body").append("div")
                .attr("class", "tooltip")
                .attr("id", "toolTipDiv")
                .style("opacity", 0);
        }

        var gHeatMap = g.selectAll(".rows")
        .data(plateData)
        .enter().append("g")
        .attr({
            "data-row": function (d, i) { return i }
        });

        var nodes = gHeatMap.selectAll("nodes")
            .data(function(d) { return d; })
            .enter()
            .append("g")
            .attr("class", "expNode")
            .on("mouseover", function () { onMouseOver(d3.select(this)) })
            .on("mouseout", function () { onMouseOut(d3.select(this)) })
            .on("click", function (element) { onClick(d3.select(this.parentNode), d3.select(this), element) });

        addShapes(nodes);
        addSymbols(nodes);

        if (displayLegend) { createLegend(g); }

        function composeMetadata() {

            //compose from plate metadata
            var plateMetaDataComp = [];
            plateMetaDataComp.push(addmetaDataType(plateMetaData.plate_BadData, plateMetaDataType.BadData));
            plateMetaDataComp.push(addmetaDataType(plateMetaData.plate_Empty, plateMetaDataType.Empty));
            plateMetaDataComp.push(addmetaDataType(plateMetaData.plate_NoGrowth, plateMetaDataType.NoGrowth));
            plateMetaDataComp.push(addmetaDataType(plateMetaData.plate_UndecidedProblem, plateMetaDataType.UndecidedProblem));

            //compose from plate data and growth metadata
            var plate = [];
            for (var i = 0; i < rows; i++) {
                var row = [];
                for (var j = 0; j < cols; j++) {
                    var metaData = findPlateMetaData(i, j, plateMetaDataComp);
                    var x = margin + (j * cellSize);
                    var y = margin + (i * cellSize);
                    var metaGt = growthMetaData.gt == undefined ? null : growthMetaData.gt[i][j];
                    var metaGtWhen = growthMetaData.gtWhen == undefined ? null : growthMetaData.gtWhen[i][j];
                    var metaYiled = growthMetaData.yld == undefined ? null : growthMetaData.yld[i][j];
                    var col = { col: j, row: i, celStartX: x, celStartY: y, celEndX: x + heatMapCelWidth, celEndY: y + heatMapCelHeight, phenotype: data[i][j], metaGT: metaGt, metaGtWhen: metaGtWhen, metaYield: metaYiled, metaType: metaData.type }
                    row.push(col);
                }
                plate.push(row);
            }
            return plate;
        }

        function findPlateMetaData(row, col, metaData) {
            for (var typeI = 0; typeI < metaData.length; typeI++) {
                var type = metaData[typeI];
                for (var itemI = 0; itemI < type.length; itemI++) {
                    var item = type[itemI];
                    if (item.row == row && item.col == col)
                        return item;
                }
            }
            return { row: row, col: col, type: plateMetaDataType.OK };
        }

        function addmetaDataType(metaDataType, typeName) {
            var dataType = [];
            var badDataElement;
            for (var k = 0; k < metaDataType[0].length; k++) {
                badDataElement = { row: metaDataType[0][k], col: metaDataType[1][k], type: typeName };
                dataType.push(badDataElement);
            }
            return dataType;
        }

        function addSymbols(nodes) {
            //Empty
            nodes
                .filter(function (d) { return d.metaType == plateMetaDataType.Empty; })
                .append("use")
                .attr({
                   "class": "plateWellSymbol",
                   "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                   "x": function (d) { return d.celStartX - 2; },
                   "y": function (d) { return d.celStartY - 0.8; },
                   "width": cellRadius * 3,
                   "height": cellRadius * 3
               });

            //UndecidedProblem
            nodes
                .filter(function (d) { return d.metaType == plateMetaDataType.UndecidedProblem; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 2; },
                    "y": function (d) { return d.celStartY; },
                    "width": cellRadius * 3,
                    "height": cellRadius * 2
                });

            //BadData
            nodes
                .filter(function (d) { return d.metaType == plateMetaDataType.BadData; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 2; },
                    "y": function (d) { return d.celStartY; },
                    "width": cellRadius * 3,
                    "height": cellRadius * 2
                });

            //NoGrowth
            nodes
                .filter(function (d) { return d.metaType == plateMetaDataType.NoGrowth; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function(d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 1; },
                    "y": function (d) { return d.celStartY; },
                    "width": cellRadius * 2.5,
                    "height": cellRadius * 2.5
                });
        }

        function addShapes(nodes) {

            //ok metadata
            nodes
            .filter(function (d) { return d.metaType == plateMetaDataType.OK; })
            .append("circle")
            .attr({
                "class": "plateWell OK",
                "fill": function (d) { return getValidColor(d.phenotype, d.metaType); },
                "id": function (d) { return "id" + d.row + "_" + d.col; },
                "fill-opacity": "1",
                "r": cellRadius,
                "cy": function (d) { return d.celStartY + cellRadius; },
                "cx": function (d) { return d.celStartX + cellRadius; },
                "data-col": function (d) { return d.col; },
                "data-meta-gt": function (d) { return d.metaGT; },
                "data-meta-gtWhen": function (d) { return d.metaGtWhen; },
                "data-meta-yield": function (d) { return d.metaYield; },
                "data-meta-type": function (d) { return d.metaType; }
            });

            //bad metadata
            nodes
            .filter(function (d) { return d.metaType != plateMetaDataType.OK; })
            .append("rect")
            .attr({
                "class": "plateWell Marked",
                "fill": function (d) { return getValidColor(d.phenotype, d.metaType); },
                "id": function (d) { return "id"+d.row + "_" + d.col; },
                "fill-opacity": "1",
                "y": function (d) { return d.celStartY; },
                "x": function (d) { return d.celStartX; },
                "width": heatMapCelWidth,
                "height": heatMapCelHeight,
                "data-col": function (d) { return d.col; },
                "data-meta-gt": function (d) { return d.metaGT; },
                "data-meta-gtWhen": function (d) { return d.metaGtWhen; },
                "data-meta-yield": function (d) { return d.metaYield; },
                "data-meta-type": function (d) { return d.metaType; }
            });
        }

        function getRowFromId(id) {
            var tmp = id.replace("id", "");
            var row = tmp.substr(0, tmp.indexOf("_"));
            return row;
        }

        function setShapes(node) {
            //ok metadata
            node
            .filter(function(d) {
                     return d.metaType == plateMetaDataType.OK;
                })
            .append("circle")
            .attr({
                "class": "plateWell OK",
                "fill": function (d) { return getValidColor(d.phenotype, d.metaType); },
                "id": function (d) { return "id" + d.row + "_" + d.col; },
                "fill-opacity": "1",
                "r": cellRadius,
                "cy": function (d) { return d.celStartY + cellRadius; },
                "cx": function (d) { return d.celStartX + cellRadius; },
                "data-col": function (d) { return d.col; },
                "data-meta-gt": function (d) { return d.metaGT; },
                "data-meta-gtWhen": function (d) { return d.metaGtWhen; },
                "data-meta-yield": function (d) { return d.metaYield; },
                "data-meta-type": function (d) { return d.metaType; }
            });

            //bad metadata
            node
            .filter(function (d) { return d.metaType != plateMetaDataType.OK; })
            .append("rect")
            .attr({
                "class": "plateWell Marked",
                "fill": function (d) { return getValidColor(d.phenotype, d.metaType); },
                "id": function (d) { return "id" + d.row + "_" + d.col; },
                "fill-opacity": "1",
                "y": function (d) { return d.celStartY; },
                "x": function (d) { return d.celStartX; },
                "width": heatMapCelWidth,
                "height": heatMapCelHeight,
                "data-col": function (d) { return d.col; },
                "data-meta-gt": function (d) { return d.metaGT; },
                "data-meta-gtWhen": function (d) { return d.metaGtWhen; },
                "data-meta-yield": function (d) { return d.metaYield; },
                "data-meta-type": function (d) { return d.metaType; }
            });
        }

        function setSymbols(node) {
            //Empty
            node
                .filter(function (d) { return d.metaType == plateMetaDataType.Empty; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 2; },
                    "y": function (d) { return d.celStartY - 0.8; },
                    "width": cellRadius * 3,
                    "height": cellRadius * 3
                });

            //UndecidedProblem
            node
                .filter(function (d) { return d.metaType == plateMetaDataType.UndecidedProblem; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 2; },
                    "y": function (d) { return d.celStartY; },
                    "width": cellRadius * 3,
                    "height": cellRadius * 2
                });

            //BadData
            node
                .filter(function (d) { return d.metaType == plateMetaDataType.BadData; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 2; },
                    "y": function (d) { return d.celStartY; },
                    "width": cellRadius * 3,
                    "height": cellRadius * 2
                });

            //NoGrowth
            node
                .filter(function (d) { return d.metaType == plateMetaDataType.NoGrowth; })
                .append("use")
                .attr({
                    "class": "plateWellSymbol",
                    "xlink:href": function (d) { return "#" + getValidSymbol(d.metaType); },
                    "x": function (d) { return d.celStartX - 1; },
                    "y": function (d) { return d.celStartY; },
                    "width": cellRadius * 2.5,
                    "height": cellRadius * 2.5
                });
        }

        function reDrawExperiment(id, mark) {
            var node = d3.select("#" + id);
            var parent = d3.select(node.node().parentNode);
            var newData = parent.data();
            newData[0].metaType = mark;
            parent.data(newData);
            parent.selectAll("*").remove();
            setShapes(parent);
            setSymbols(parent);
        }

        function setExperiment(id) {
            var well = d3.select("#" + id);
            if (well.empty())
                return;

            var row = getRowFromId(id);
            var col = well.attr("data-col");
            var metaDataGt = well.attr("data-meta-gt");
            var metaDataGtWhen = well.attr("data-meta-gtWhen");
            var metaDataYield = well.attr("data-meta-yield");
            var coord = row + "," + col;
            var coordinate = "[" + coord + "]";
            var phenotype = 0;
            var fmt = d3.format(numberFormat);
            well.attr("",
                function (d) { phenotype = fmt(d.phenotype) });
            var exp = { id: id, coord: coord, metaDataGt: metaDataGt, metaDataGtWhen: metaDataGtWhen, metaDataYield: metaDataYield, phenotype: phenotypeName };
            //deselect preavius selections
            var sel = g.selectAll("." + classExperimentSelected);
            sel.classed(classExperimentSelected, false);
            sel.attr({ "stroke-width": 0 });
            //new selection
            var newSel = well;
            newSel.classed(classExperimentSelected, true);
            newSel.attr({
                "stroke": "black",
                "stroke-width": 3
            });
            //trigger click and send coordinate
            toolTipDiv.transition().duration(0).style("opacity", 0);
            dispatch[dispatcherSelectedExperiment](exp);
        };

        function onClick(rowNode, thisNode, element) {
            var row = rowNode.attr("data-row");
            var well = thisNode.select(".plateWell");
            var col = well.attr("data-col");
            var id = well.attr("id");
            var metaDataGt = well.attr("data-meta-gt");
            var metaDataGtWhen = well.attr("data-meta-gtWhen");
            var metaDataYield = well.attr("data-meta-yield");
            var coord = row + "," + col;
            var coordinate = "[" + coord + "]";
            var phenotype = 0;
            var fmt = d3.format(numberFormat);
            well.attr("",
                function (d) { phenotype = fmt(d.phenotype) });
            var exp = { id: id, coord: coord, metaDataGt: metaDataGt, metaDataGtWhen: metaDataGtWhen, metaDataYield: metaDataYield, phenotype: phenotypeName };
            //deselect preavius selections
            var sel = g.selectAll("." + classExperimentSelected);
            sel.classed(classExperimentSelected, false);
            sel.attr({"stroke-width": 0});
            //new selection
            var newSel = well;
            newSel.classed(classExperimentSelected, true);
            newSel.attr({
                "stroke": "black",
                "stroke-width" :3
            });
            //trigger click and send coordinate
            toolTipDiv.transition().duration(0).style("opacity", 0);
            window.qc.actions.setQualityIndex(window.qc.selectors.getQIndexFromPosition(
                parseInt(row, 10),
                parseInt(col, 10),
            ));
        }

        function onMouseOut(node) {
            node.select(".plateWell").attr("fill", function (d) { return getValidColor(d.phenotype, d.metaType); });
            node.select(".plateWellSymbol").attr("fill", "black");

            toolTipDiv.transition()
                .duration(0)
                .style("opacity", 0);
        }

        function onMouseOver(node) {
            if (markingSelection == true) return;


            var size = 50;

            var fmt = d3.format(numberFormat);
            toolTipDiv.transition()
                    .duration(0)
                    .style("opacity", .9);

            node.select(".plateWellSymbol")
                .attr("fill", "white");

            toolTipDiv.html("");
            var toolTipIcon = toolTipDiv.append("svg")
                .attr({
                    "width": size,
                    "height": size,
                    "style": "float: left"
                });

            addSymbolsToSGV(toolTipIcon);

            node.select(".plateWell")
                .attr("fill", "black")
                .attr("", function (d) {

                    toolTipIcon.append("use")
                    .attr({
                        "class": "toolTipSymbol",
                        "xlink:href": function() { return "#" + getValidSymbol(d.metaType); },
                        "x": 0,
                        "y": 5,
                        "width": size,
                        "height": size,
                        "fill": "white"
                    });

                    toolTipDiv.append("div").text(phenotypeName);
                    toolTipDiv.append("div").text(fmt(d.phenotype));
                    toolTipDiv
                        .style("left", (d3.event.pageX) + "px")
                        .style("top", (d3.event.pageY - size) + "px");
                });
        }
    }

    function getValidColor(phenotype, dataType) {
        var color;
        if (dataType === plateMetaDataType.OK)
            color = colorScale(phenotype);
        else
            color = "white";
        return color;
    }

    function createLegend(container) {

        var startX = margin + (cols * cellSize) + legendMargin;
        var startY = margin;
        var heatMaphight = (rows * cellSize) ;
        var gLegendScale = container.append("g");

        var gradient = gLegendScale
            .append("linearGradient")
            .attr({
                "y1": "0%",
                "y2": "100%",
                "x1": "0",
                "x2": "0",
                "id": "gradient"
            });

        gradient
            .append("stop")
            .attr({
                "offset": "0%",
                "stop-color": colorSchema[2]
            });

        gradient
            .append("stop")
            .attr({
                "offset": "50%",
                "stop-color": colorSchema[1]
            });

        gradient
            .append("stop")
                .attr({
                    "offset": "100%",
                    "stop-color": colorSchema[0]
                });

        gLegendScale.append("rect")
            .attr({
                y: startY,
                x: startX,
                width: legendWidth,
                height: heatMaphight
            }).style({
                fill: "url(#gradient)",
                "stroke-width": 2,
                "stroke": "black"
            });

        var gLegendaxis = container.append("g").classed("HeatMapLegendAxis", true);

        var dom = [phenotypeMax, phenotypeMin];
        //d3.extent(data[0]).reverse()
        var legendScale = d3.scale.linear()
        .domain(dom)
        .rangeRound([startY, heatMaphight])
        .nice();

        var gradAxis = d3.svg.axis()
            .scale(legendScale)
            .orient("right")
            .tickSize(10)
            .ticks(10)
            .tickFormat(d3.format(numberFormat));

        gradAxis(gLegendaxis);
        gLegendaxis.attr({
            "transform": "translate(" + (startX + legendWidth-9) + ", " + (margin/2) + ")"
        });
        gLegendaxis.selectAll("path").style({ fill: "none", stroke: "#000" });
        gLegendaxis.selectAll("line").style({ stroke: "#000" });
    }

    heatmap.data = function(value) {
        if (!arguments.length) return data;
        data = value;
        cols = data[0].length;
        rows = data.length;
        phenotypeMin = d3.min(data, function (array) { return d3.min(array) });
        phenotypeMax = d3.max(data, function(array) { return d3.max(array) });
        phenotypeMean = d3.mean(data, function(array) { return d3.mean(array) });
        return heatmap;
    }

    heatmap.phenotypeName = function (value) {
        if (!arguments.length) return phenotypeName;
        phenotypeName = value;
        return heatmap;
    }

    heatmap.growthMetaData = function (value) {
        if (!arguments.length) return growthMetaData;
        growthMetaData = value;
        return heatmap;
    }

    heatmap.plateMetaData = function (value) {
        if (!arguments.length) return plateMetaData;
        plateMetaData = value;
        return heatmap;
    }

    heatmap.cellSize = function(value) {
        if (!arguments.length) return cellSize;
        cellSize = value;
        return heatmap;
    }

    heatmap.cellRadius = function(value) {
        if (!arguments.length) return cellRadius;
        cellRadius = value;
        heatMapCelHeight = value * 2;
        heatMapCelWidth = value * 2;
        return heatmap;
    }

    heatmap.colorScale = function(value) {
        if (!arguments.length) return colorScale;
        colorScale = value;
        return heatmap;
    }

    heatmap.colorSchema = function (value) {
        if (!arguments.length) return colorSchema;
        colorSchema = value;
        return heatmap;
    }

    heatmap.setColorScale = function(value) {

        if (typeof value === "undefined" || value === null) {
            throw "colorSchema isundefined";
        }
        if (typeof data === "undefined" || data === null) {
            throw "data is not set!";
        }

        colorSchema = value;
        var cs = d3.scale.linear()
            .domain([phenotypeMin, phenotypeMean, phenotypeMax])
            .range([colorSchema[0], colorSchema[1], colorSchema[2]]);
        colorScale = cs;
    }

    heatmap.margin = function (value) {
        if (!arguments.length) return margin;
        margin = value;
        return heatmap;
    }

    heatmap.displayLegend = function (value) {
        if (!arguments.length) return displayLegend;
        displayLegend = value;
        return heatmap;
    }

    heatmap.legendMargin = function (value) {
        if (!arguments.length) return legendMargin;
        legendMargin = value;
        return heatmap;
    }

    heatmap.legendWidth = function (value) {
        if (!arguments.length) return legendWidth;
        legendWidth = value;
        return heatmap;
    }

    heatmap.dispatch2 = function (value) {
        if (!arguments.length) return dispatch2;
        dispatch2 = value;
        return heatmap;
    }

    return d3.rebind(heatmap, dispatch, "on");
}
