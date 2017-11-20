import PropTypes from 'prop-types';
import React from 'react';

import { GetFixturePlates } from '../api';
import CCCEditor from '../components/CCCEditor';


export default class CCCEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            plates: [],
            currentPlate: null,
            ready: false,
        };
        this.handleFinishUpload = this.handleFinishUpload.bind(this);
        this.handleFinishPlate = this.handleFinishPlate.bind(this);
    }

    componentDidMount() {
        GetFixturePlates(this.props.fixtureName)
            .then(plates => {
                this.setState({ ready: true, platesPerImage: plates.length });
            });
    }

    handleFinishUpload(newImage) {
        const plates = [...this.state.plates];
        for (let i = 1 ; i <= this.state.platesPerImage ; i++) {
            const newPlate = {
                imageId: newImage.id,
                imageName: newImage.name,
                plateId: i,
            };
            plates.push(newPlate);
        }
        const currentPlate =
            this.state.currentPlate == null
            ? this.state.plates.length
            : this.state.currentPlate;
        this.setState({
            plates,
            currentPlate,
        });
    }

    handleFinishPlate() {
        const currentPlate =
            this.state.currentPlate < this.state.plates.length - 1
            ? this.state.currentPlate + 1
            : null;
        this.setState({ currentPlate });
    }

    render() {
        return (
            <CCCEditor
                cccId={this.props.cccId}
                accessToken={this.props.accessToken}
                pinFormat={this.props.pinFormat}
                fixtureName={this.props.fixtureName}
                plates={this.state.plates}
                currentPlate={this.state.currentPlate}
                onFinishUpload={this.handleFinishUpload}
                onFinishPlate={this.handleFinishPlate}
                ready={this.state.ready}
            />
        );
    }
}

CCCEditorContainer.propTypes = {
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    fixtureName: PropTypes.string.isRequired,
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
};
