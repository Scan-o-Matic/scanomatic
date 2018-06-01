// @flow
import $ from 'jquery';

export default class DrawCurvesIntegration {
    handleUpdate: () => void;
    hasDrawn: boolean;
    plate: number;
    row: number;
    col: number;

    constructor() {
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    shouldDraw({ plate, row, col } : { plate: number, row: number, col: number }) : boolean {
        if (!this.hasDrawn) return true;
        return this.plate !== plate || this.row !== row || this.col !== col;
    }

    handleUpdate() {
        const selector = '#graph';
        const focus = window.qc.selectors.getFocus();
        const plate = window.qc.selectors.getPlate();
        if (!focus) {
            $(selector).hide();
            return;
        }
        if (!this.shouldDraw({ plate, ...focus })) return;

        const raw = window.qc.selectors.getRawCurve(plate, focus.row, focus.col);
        const smooth = window.qc.selectors.getSmoothCurve(plate, focus.row, focus.col);
        const time = window.qc.selectors.getTimes(plate);
        if (!raw || !smooth || !smooth) {
            this.hasDrawn = false;
            $(selector).hide();
            return;
        }

        const well = `#id${focus.row}_${focus.col}`;
        // This is horrible, but seems only way to access all needed data
        // in  the legacy code.
        const data = $(well)[0].__data__;
        window.DrawCurves(
            selector,
            time,
            raw,
            smooth,
            data.metaGT,
            data.metaGtWhen,
            data.metaYield,
        );
        this.hasDrawn = true;
        this.plate = plate;
        this.row = focus.row;
        this.col = focus.col;
        $(selector).show();
    }
}
