// @flow
import $ from 'jquery';
import Bridge from './bridge';


export default class QIndexAndSelectionIntegration {
    bridge: Bridge;
    handleUpdate: () => void;

    constructor(bridge: Bridge) {
        this.bridge = bridge;
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    handleUpdate() {
        const focus = Object.assign(
            {},
            this.bridge.selectors.getFocus(),
            { plate: this.bridge.selectors.getPlate() },
        );

        const well = $(`#id${focus.row}_${focus.col}`)[0];
        if (!well) return;

        $('.selected-experiment').removeClass('selected-experiment');
        $(`#id${focus.row}_${focus.col}`).addClass('selected-experiment');
        $('#qIndexCurrent').val(focus.idx + 1);
    }
}
