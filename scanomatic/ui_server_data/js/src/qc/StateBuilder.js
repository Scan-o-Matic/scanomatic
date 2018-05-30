import type { State, TimeSeries, Plate, Settings } from './state';
import { getUpdated2DArrayCopy } from './helpers';

export default class StateBuilder {
    plate: Plate;
    settings: Settings;

    constructor() {
        this.plate = { number: 0 };
        this.settings = {};
    }

    setProject(project: string) {
        this.settings = { project };
        this.plate = { number: 0 };
        return this;
    }

    setPlate(plate: number) {
        this.plate = { number: plate };
        return this;
    }

    setPinning(rows: number, cols: number) {
        this.plate = Object.assign({}, this.plate, { pinning: { rows, cols } });
        return this;
    }

    setRawCurveData(plate: number, row: number, col: number, data: TimeSeries) {
        if (plate !== this.plate.number) return this;
        this.plate.raw = getUpdated2DArrayCopy(this.plate.raw, row, col, data);
        return this;
    }

    setSmoothCurveData(plate: number, row: number, col: number, data: TimeSeries) {
        if (plate !== this.plate.number) return this;
        this.plate.smooth = getUpdated2DArrayCopy(this.plate.smooth, row, col, data);
        return this;
    }

    setTimes(plate: number, times: TimeSeries) {
        if (plate !== this.plate.number) return this;
        this.plate.times = times;
        return this;
    }

    build() : State {
        return {
            plate: this.plate,
            settings: this.settings,
        };
    }
}
