// @flow
import $ from 'jquery';
import Bridge from './bridge';

function updateQIndexLabel({ idx }) {
    $('#qIndexCurrent').text(idx + 1);
}


export default class QIndexAndSelectionIntegration {
    bridge: Bridge;
    plate: number;
    row: number;
    col: number;
    handleUpdate: () => void;

    constructor(bridge: Bridge) {
        this.bridge = bridge;
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    shouldSync({ plate, row, col } : { plate: number, row: number, col: number }) : boolean {
        return this.plate !== plate || this.row !== row || this.col !== col;
    }

    handleUpdate() {
        const focus = Object.assign(
            {},
            this.bridge.selectors.getFocus(),
            { plate: this.bridge.selectors.getPlate() },
        );
        updateQIndexLabel(focus);
        if (!this.shouldSync(focus)) return;
        this.plate = focus.plate;
        this.row = focus.row;
        this.col = focus.col;
        dispatch.setExp(`id${focus.row}_${focus.col}`);
    }
}
