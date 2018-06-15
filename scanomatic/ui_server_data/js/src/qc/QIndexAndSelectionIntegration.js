// @flow
import $ from 'jquery';
import Bridge from './bridge';


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
        $('#qIndexCurrent').text(focus.idx + 1);
        if (!this.shouldSync(focus)) return;
        this.plate = focus.plate;
        this.row = focus.row;
        this.col = focus.col;

        // I don't know exactly how and where this function ends up in the
        // global scope, but there it is. It's not a property of window or
        // document so I don't know how I should deal with the lint error.
        dispatch.setExp(`id${focus.row}_${focus.col}`);
    }
}
