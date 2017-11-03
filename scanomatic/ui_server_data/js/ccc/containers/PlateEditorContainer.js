import PropTypes from 'prop-types';
import React from 'react';

import PlateEditor from '../components/PlateEditor';


export default class PlateEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            step: 'gridding',
            selectedColony: null,
        };
        this.handleGriddingFinish = this.handleGriddingFinish.bind(this);
        this.handleColonyFinish = this.handleColonyFinish.bind(this);
    }

    handleGriddingFinish() {
        this.setState({
            step: 'colony',
            selectedColony: { row: 0, col: 0 },
        });
    }

    handleColonyFinish() {
        const { row, col } = this.state.selectedColony;
        const [nCol, nRow] = this.props.pinFormat;
        if (col < nCol - 1) {
            this.setState({ selectedColony: { row, col: col + 1 } });
        } else if (row < nRow - 1) {
            this.setState({ selectedColony: { row: row + 1, col: 0 } });
        }
    }

    render() {
        return (
            <PlateEditor
                step={this.state.step}
                onGriddingFinish={this.handleGriddingFinish}
                onColonyFinish={this.handleColonyFinish}
                cccId={this.props.cccId}
                imageId={this.props.imageId}
                plateId={this.props.plateId}
                accessToken={this.props.accessToken}
                pinFormat={this.props.pinFormat}
                selectedColony={this.state.selectedColony}
            />
        );
    }
}

PlateEditorContainer.propTypes = {
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
    imageId: PropTypes.string.isRequired,
    plateId: PropTypes.string.isRequired,
};
