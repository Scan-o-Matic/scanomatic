// @flow
import type { State, UpdateStatus, Scanners, Experiments } from './state';

export default class StateBuilder {
    experiments: Experiments;
    scanners: Scanners;
    updateStatus: UpdateStatus;

    constructor() {
        this.experiments = [];
        this.scanners = [];
        this.updateStatus = {};
    }

    populateScanners() {
        this.scanners = [
            {
                name: 'Hollowborn Heron',
                id: 'scanner001',
                isOnline: true,
                isFree: true,
            },
            {
                name: 'Eclectic Eevee',
                id: 'scanner002',
                isOnline: false,
                isFree: true,
            },
        ];
        this.updateStatus.scanners = new Date();
        return this;
    }

    populateExperiments() {
        this.experiments = [
            {
                name: 'A quick test',
                description: 'Manual testing FTW!',
                duration: 55,
                interval: 1,
                scannerId: 'scanner001',
                started: new Date(),
            },
            {
                name: 'Sun and Moon',
                description: 'Yellow',
                duration: 55,
                interval: 1,
                scannerId: 'scanner001',
                started: new Date(),
                stopped: new Date(),
                end: new Date(),
            },
        ];
        this.updateStatus.experiments = new Date();
        return this;
    }

    build(): State {
        return {
            experiments: this.experiments,
            scanners: this.scanners,
            updateStatus: this.updateStatus,
        };
    }
}
