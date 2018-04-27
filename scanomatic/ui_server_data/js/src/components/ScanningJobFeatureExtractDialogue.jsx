import PropTypes from 'prop-types';
import React from 'react';


class ScanningJobFeatureExtractDialogue extends React.Component {
    constructor() {
        super();
        this.state = { keepQC: false };
        this.handleExtractFeatures = this.handleExtractFeatures.bind(this);
        this.handleKeepQC = this.handleKeepQC.bind(this);
    }

    handleExtractFeatures() {
        this.props.onExtractFeatures(this.state.keepQC);
    }

    handleKeepQC(e) {
        this.setState({ keepQC: e.target.checked });
    }

    render() {
        const { onCancel } = this.props;
        const { keepQC } = this.state;
        return (
            <div
                className="jumbotron scanning-job-stop-dialogue"
            >
                <h1>Feature Extraction</h1>
                <h3>This will remove previous extracted phenotyes</h3>
                <p className="text-justify">
                    Feature extraction extracts information from the growth-curves.
                    If you have already done quality control of these phenotypes,
                    then you can select &ldquo;Attempt to keep QC&rdquo; below.
                    If the existing feature extraction is compatible with the new,
                    the QC marks of the curves will be transferred to the new features.
                </p>
                <div className="checkbox">
                    <label>
                        <input
                            className="keep-qc"
                            checked={keepQC}
                            type="checkbox"
                            onChange={this.handleKeepQC}
                        /> Attempt to keep QC
                    </label>
                </div>
                <p>
                    <button
                        className="btn btn-primary btn-lg feature-extract-button"
                        onClick={this.handleExtractFeatures}
                    >
                        Extract Features
                    </button>
                    <button
                        className="btn btn-default btn-lg cancel-button"
                        onClick={onCancel}
                    >
                        Cancel
                    </button>
                </p>
            </div>
        );
    }
}

ScanningJobFeatureExtractDialogue.propTypes = {
    onCancel: PropTypes.func.isRequired,
    onExtractFeatures: PropTypes.func.isRequired,
};

export default ScanningJobFeatureExtractDialogue;
