import PropTypes from 'prop-types';
import React from 'react';

import ColonyEditor from '../components/ColonyEditor';
import { SetColonyCompressionV2, SetColonyDetection } from '../api';

export default class ColonyEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.handleSet = this.handleSet.bind(this);
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    componentDidMount() {
        const { scope, ccc, image, plate, row, col, accessToken } = this.props;
        SetColonyDetection(
            scope, ccc, image, plate, accessToken, row, col,
            this.handleColonyDetectionSuccess.bind(this),
            () => {},
        );
    }

    componentWillReceiveProps(newProps) {
        this.setState({ colonyData: null });
        const { scope, ccc, image, plate, row, col, accessToken } = newProps;
        SetColonyDetection(
            scope, ccc, image, plate, accessToken, row, col,
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
        const { scope, ccc, image, plate, row, col, accessToken } = this.props;
        const { colonyData } = this.state;
        SetColonyCompressionV2(
            scope, ccc, image, plate, accessToken, colonyData, row, col,
            () => { this.props.onFinish && this.props.onFinish() },
            (data) => { alert(`Set Colony compression Error: ${data.reason}`); }
        );
    }

    render() {
        if (!this.state.colonyData) {
            return null;
        }
        return (
            <ColonyEditor
                data={this.state.colonyData}
                onSet={this.handleSet}
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
    scope: PropTypes.object.isRequired,
    onFinish: PropTypes.func,
};
