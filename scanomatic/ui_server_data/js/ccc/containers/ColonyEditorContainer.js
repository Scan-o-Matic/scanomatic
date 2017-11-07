import PropTypes from 'prop-types';
import React from 'react';

import ColonyEditor from '../components/ColonyEditor';
import { SetColonyCompression, SetColonyDetection } from '../api';

export default class ColonyEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.handleSet = this.handleSet.bind(this);
        this.handleSkip = this.handleSkip.bind(this);
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    componentDidMount() {
        this.getColonyData(this.props);
    }

    componentWillReceiveProps(newProps) {
        this.setState({ colonyData: null });
        this.getColonyData(newProps);
    }

    getColonyData({ ccc, image, plate, row, col, accessToken }) {
        SetColonyDetection(
            ccc, image, plate, accessToken, row, col,
            this.handleColonyDetectionSuccess.bind(this),
            () => {},
        );
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
        const { ccc, image, plate, row, col, accessToken } = this.props;
        const { colonyData } = this.state;
        SetColonyCompression(
            ccc, image, plate, accessToken, colonyData, row, col,
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
    plate: PropTypes.string.isRequired,
    row: PropTypes.number.isRequired,
    col: PropTypes.number.isRequired,
    onFinish: PropTypes.func,
};
