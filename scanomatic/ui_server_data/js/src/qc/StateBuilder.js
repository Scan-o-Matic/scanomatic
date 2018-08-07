// @flow

import type {
    State, TimeSeries, PlateOfTimeSeries, Plate, Settings, QualityIndexQueue,
    PlateValueArray, PlateCoordinatesArray, Phenotype, QCMarksMap,
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

    setPhenotype(phenotype: Phenotype) {
        if (!this.settings.project) return this;
        this.settings = Object.assign({}, this.settings, { phenotype });
        return this;
    }

    setPlatePhenotypeData(
        phenotype: Phenotype,
        phenotypes: PlateValueArray,
        qc: QCMarksMap,
    ) {
        const nextPhenotypes = new Map(this.plate.phenotypes);
        const nextQC = new Map(this.plate.qcmarks);
        nextPhenotypes.set(phenotype, phenotypes);
        nextQC.set(phenotype, qc);
        this.plate = Object.assign({}, this.plate, { phenotypes: nextPhenotypes, qcmarks: nextQC });
        return this;
    }

    setPlate(plate: number) {
        this.plate = { number: plate, qIndex: 0 };
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
