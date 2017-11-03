import PropTypes from 'prop-types';
import React from 'react';

import Gridding from '../components/Gridding';
import { SetGridding } from '../api';

export default class GriddingContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            rowOffset: 4,
            colOffset: 0,
            status: 'loading',
        };
        this.handleRowOffsetChange = this.handleRowOffsetChange.bind(this);
        this.handleColOffsetChange = this.handleColOffsetChange.bind(this);
        this.handleRegrid = this.handleRegrid.bind(this);
    }

    handleRowOffsetChange(newRowOffset) {
        this.setState({ rowOffset: newRowOffset });
    }

    handleColOffsetChange(newColOffset) {
        this.setState({ colOffset: newColOffset });
    }

    componentDidMount() {
        this.updateGrid([0, 0]);
    }

    updateGrid(offsets) {
        this.setState({
            alert: 'Calculating Gridding ... please wait ...!',
            status: 'loading',
        });
        const { scope, cccId, imageId, plateId, pinFormat, accessToken } = this.props;
        SetGridding(
            scope, cccId, imageId, plateId, pinFormat, offsets, accessToken,
            this.handleSetGriddingSuccess.bind(this),
            this.handleSetGriddingError.bind(this),
        );
    }

    handleSetGriddingSuccess(data) {
        this.setState({
            grid: data.grid,
            alert: 'Gridding was succesful!',
            status: 'success',
        });
    }

    handleSetGriddingError(data) {
        this.setState({
            grid: data.grid,
            alert: `Gridding was unsuccesful. Reason: '${data.reason}'. Please enter Offset and retry!`,
            status: 'error',
        });
    }

    handleRegrid() {
        this.updateGrid([this.state.rowOffset, this.state.colOffset]);
    }

    render() {
        return (
            <Gridding
                alert={this.state.alert}
                status={this.state.status}
                cccId={this.props.cccId}
                imageId={this.props.imageId}
                plateId={this.props.plateId}
                colOffset={this.state.colOffset}
                rowOffset={this.state.rowOffset}
                onColOffsetChange={this.handleColOffsetChange}
                onRowOffsetChange={this.handleRowOffsetChange}
                onRegrid={this.handleRegrid}
                onNext={this.props.onFinish}
                image={new Image}
                grid={this.state.grid}
                selectedColony={this.props.selectedColony}
            />
        );
    }
}

GriddingContainer.propTypes = {
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
    imageId: PropTypes.string.isRequired,
    plateId: PropTypes.string.isRequired,
    scope: PropTypes.object.isRequired,
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    onFinish: PropTypes.func.isRequired,
    selectedColony: PropTypes.shape({
        row: PropTypes.number,
        col: PropTypes.number,
    }),
};
