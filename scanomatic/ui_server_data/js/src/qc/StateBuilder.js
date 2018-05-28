import type { State, TimeSeries, Plate, Settings } from './state';
import { getUpdated2DArrayCopy } from './helpers';

export default class StateBuilder {
    plate: Plate;
    settings: Settings;

    constructor() {
        this.plate = { number: 1 };
        this.settings = {};
    }

    setPlate(plate: number) {
        this.plate = { number: plate };
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

    build() : State {
        return {
            plate: this.plate,
            settings: this.settings,
        };
    }
}
