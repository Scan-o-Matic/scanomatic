import PropTypes from 'prop-types';
import React from 'react';

import ColonyEditor from '../components/ColonyEditor';
import { SetColonyCompression, SetColonyDetection } from '../api';

export default class ColonyEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            colonyData: null,
            cellCount: null,
            cellCountError: false,
        };
        this.handleCellCountChange = this.handleCellCountChange.bind(this);
        this.handleSet = this.handleSet.bind(this);
        this.handleSkip = this.handleSkip.bind(this);
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    componentDidMount() {
        this.getColonyData(this.props);
    }

    componentWillReceiveProps(newProps) {
        this.setState({
            cellCount: null,
            cellCountError: false,
            colonyData: null,
        });
        this.getColonyData(newProps);
    }

    getColonyData({ ccc, image, plateId, row, col, accessToken }) {
        SetColonyDetection(
            ccc, image, plateId, accessToken, row, col,
            this.handleColonyDetectionSuccess.bind(this),
            () => {},
        );
    }

    handleCellCountChange(cellCount) {
        this.setState({ cellCount, cellCountError: cellCount < 0 });
    }

    handleColonyDetectionSuccess(data) {
        this.setState({ colonyData: {
            image: data.image,
            imageMin: data.image_min,
            imageMax: data.image_max,
            blob: data.blob,
            background: data.background,
        }});
    }

    handleUpdate(data) {
        const colonyData = Object.assign({}, this.state.colonyData, data);
        this.setState({ colonyData });
    }

    handleSet() {
        if (this.state.cellCount == null || this.state.cellCountError) {
            this.setState({ cellCountError: true });
            return;
        }
        const { ccc, image, plateId, row, col, accessToken } = this.props;
        const { cellCount, colonyData } = this.state;
        SetColonyCompression(
            ccc, image, plateId, accessToken, colonyData, cellCount, row, col,
            () => { this.props.onFinish && this.props.onFinish() },
            (data) => { alert(`Set Colony compression Error: ${data.reason}`); }
        );
    }

    handleSkip() {
        this.props.onFinish && this.props.onFinish();
    }

    render() {
        if (!this.state.colonyData) {
            return null;
        }
        return (
            <ColonyEditor
                data={this.state.colonyData}
                cellCount={this.state.cellCount}
                cellCountError={this.state.cellCountError}
                onCellCountChange={this.handleCellCountChange}
                onSet={this.handleSet}
                onSkip={this.handleSkip}
                onUpdate={this.handleUpdate}
            />
        );
    }
}

ColonyEditorContainer.propTypes = {
    accessToken: PropTypes.string.isRequired,
    ccc: PropTypes.string.isRequired,
    image: PropTypes.string.isRequired,
    plateId: PropTypes.number.isRequired,
    row: PropTypes.number.isRequired,
    col: PropTypes.number.isRequired,
    onFinish: PropTypes.func,
};
