// @flow
import type { State, UpdateStatus, Scanners, Experiments } from './state';

export default class StateBuilder {
    experiments: Experiments;
    scanners: Scanners;
    updateStatus: UpdateStatus;

    constructor() {
        this.experiments = [];
        this.scanners = [];
        this.updateStatus = {
            scanners: null,
            experiments: null,
        };
    }

    populateScanners() {
        this.scanners = [
            {
                name: 'Hollowborn Heron',
                id: 'scanner001',
                isOnline: true,
            },
            {
                name: 'Eclectic Eevee',
                id: 'scanner002',
                isOnline: false,
            },
        ];
        this.updateStatus = Object.assign({}, this.updateStatus, { scanners: new Date() });
        return this;
    }

    populateExperiments() {
        this.experiments = [
            {
                id: 'aaa',
                name: 'A quick test',
                scannerId: 'scanner001',
                started: new Date().getTime(),
                stopped: null,
                end: null,
            },
            {
                id: 'bbb',
                name: 'Sun and Moon',
                scannerId: 'scanner001',
                started: new Date().getTime(),
                stopped: new Date().getTime(),
                end: new Date().getTime(),
            },
        ];
        this.updateStatus = Object.assign({}, this.updateStatus, { experiments: new Date() });
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
