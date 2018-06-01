// @flow

import type { State, TimeSeries, PlateOfTimeSeries, Plate, Settings } from './state';

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

    setFocus(plate: number, row: number, col: number) {
        if (plate !== this.plate.number) return this;
        this.plate.focus = { row, col };
        return this;
    }

    setPlateGrowthData(
        plate: number,
        times: TimeSeries,
        raw: PlateOfTimeSeries,
        smooth: PlateOfTimeSeries,
    ) {
        if (plate !== this.plate.number) return this;
        this.plate.times = times;
        this.plate.raw = raw;
        this.plate.smooth = smooth;
        return this;
    }

    build() : State {
        return {
            plate: this.plate,
            settings: this.settings,
        };
    }
}
