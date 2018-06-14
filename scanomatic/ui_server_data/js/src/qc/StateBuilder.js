// @flow

import type {
    State, TimeSeries, PlateOfTimeSeries, Plate, Settings, QualityIndexQueue,
} from './state';

export default class StateBuilder {
    plate: Plate;
    settings: Settings;

    constructor() {
        this.plate = { number: 0, qIndex: 0 };
        this.settings = {};
    }

    setProject(project: string) {
        this.settings = { project };
        this.plate = { number: 0, qIndex: 0 };
        return this;
    }

    setPlate(plate: number) {
        this.plate = { number: plate, qIndex: 0 };
        return this;
    }

    setFocus(plate: number, row: number, col: number) {
        if (plate !== this.plate.number) return this;
        this.plate = Object.assign({}, this.plate, { focus: { row, col } });
        return this;
    }

    setPlateGrowthData(
        plate: number,
        times: TimeSeries,
        raw: PlateOfTimeSeries,
        smooth: PlateOfTimeSeries,
    ) {
        if (plate !== this.plate.number) return this;
        this.plate = Object.assign({}, this.plate, { times, raw, smooth });
        return this;
    }

    setQualityIndexQueue(queue: QualityIndexQueue) {
        this.plate = Object.assign({}, this.plate, { qIndexQueue: queue });
        return this;
    }

    setQualityIndex(index: number) {
        if (!this.plate.qIndexQueue) return this;
        this.plate = Object.assign({}, this.plate, { qIndex: index });
        return this;
    }

    build() : State {
        return {
            plate: this.plate,
            settings: this.settings,
        };
    }
}
