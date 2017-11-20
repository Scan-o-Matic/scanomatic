import PropTypes from 'prop-types';
import React from 'react';

import PlateEditor from '../components/PlateEditor';
import { SetGrayScaleTransform, SetGridding } from '../api';


export default class PlateEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            step: 'pre-processing',
            rowOffset: 0,
            colOffset:0,
            selectedColony: { row: 0, col: 0 },
        };
        this.handleColOffsetChange = this.handleColOffsetChange.bind(this);
        this.handleColonyFinish = this.handleColonyFinish.bind(this);
        this.handleClickNext = this.handleClickNext.bind(this);
        this.handleRegrid = this.handleRegrid.bind(this);
        this.handleRowOffsetChange = this.handleRowOffsetChange.bind(this);
    }

    componentDidMount() {
        const { cccId, imageId, plateId, accessToken } = this.props;
        SetGrayScaleTransform(cccId, imageId, plateId, accessToken)
            .then(this.handleSetGrayScaleTransformSuccess.bind(this));
    }

    handleSetGrayScaleTransformSuccess() {
        this.setState({ step: 'gridding', griddingLoading: true });
        SetGridding(
            this.props.cccId, this.props.imageId, this.props.plateId,
            this.props.pinFormat, [0, 0], this.props.accessToken,
        ).then(
            this.handleSetGriddingSuccess.bind(this),
            this.handleSetGriddingError.bind(this),
        );
    }

    handleSetGriddingSuccess({ grid }) {
        this.setState({ griddingLoading: false, grid });
    }

    handleSetGriddingError({ reason, grid }) {
        this.setState({ griddingLoading: false, griddingError: reason, grid });
    }

    handleRegrid() {
        this.setState({ step: 'gridding', griddingLoading: true, griddingError: null });
        SetGridding(
            this.props.cccId, this.props.imageId, this.props.plateId,
            this.props.pinFormat, [this.state.rowOffset, this.state.colOffset],
            this.props.accessToken,
        ).then(
            this.handleSetGriddingSuccess.bind(this),
            this.handleSetGriddingError.bind(this),
        );
    }

    handleRowOffsetChange(rowOffset) {
        this.setState({ rowOffset });
    }

    handleColOffsetChange(colOffset) {
        this.setState({ colOffset });
    }

    handleClickNext() {
        if (this.state.step === 'gridding') {
            this.setState({ step: 'colony-detection' });
        } else if (this.state.step === 'colony-detection') {
            this.setState({ step: 'done' });
            this.props.onFinish && this.props.onFinish()
        }
    }

    handleColonyFinish() {
        const { row, col } = this.state.selectedColony;
        const [nCol, nRow] = this.props.pinFormat;
        if (col < nCol - 1) {
            this.setState({ selectedColony: { row, col: col + 1 } });
        } else if (row < nRow - 1) {
            this.setState({ selectedColony: { row: row + 1, col: 0 } });
        } else {
            this.props.onFinish && this.props.onFinish()
        }
    }

    render() {
        return (
            <PlateEditor
                accessToken={this.props.accessToken}
                cccId={this.props.cccId}
                colOffset={this.state.colOffset}
                grid={this.state.grid}
                griddingError={this.state.griddingError}
                griddingLoading={this.state.griddingLoading}
                imageId={this.props.imageId}
                imageName={this.props.imageName}
                onColOffsetChange={this.handleColOffsetChange}
                onColonyFinish={this.handleColonyFinish}
                onClickNext={this.handleClickNext}
                onRegrid={this.handleRegrid}
                onRowOffsetChange={this.handleRowOffsetChange}
                pinFormat={this.props.pinFormat}
                plateId={this.props.plateId}
                rowOffset={this.state.rowOffset}
                selectedColony={this.state.selectedColony}
                step={this.state.step}
                collapse={this.props.collapse}
            />

        );
    }
}

PlateEditorContainer.propTypes = {
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
    collapse: PropTypes.bool,
    imageId: PropTypes.string.isRequired,
    imageName: PropTypes.string.isRequired,
    onFinish: PropTypes.func,
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    plateId: PropTypes.number.isRequired,
};
