import PropTypes from 'prop-types';
import Duration from './Duration';


const pinningFormat = PropTypes.shape({
    name: PropTypes.string.isRequired,
    nRows: PropTypes.number.isRequired,
    nCols: PropTypes.number.isRequired,
});

const cccMetadata = PropTypes.shape({
    id: PropTypes.string.isRequired,
    accessToken: PropTypes.string.isRequired,
    species: PropTypes.string.isRequired,
    reference: PropTypes.string.isRequired,
    fixtureName: PropTypes.string.isRequired,
    pinningFormat: pinningFormat.isRequired,
});

const scannerShape = {
    identifier: PropTypes.string,
    name: PropTypes.string.isRequired,
    owned: PropTypes.bool.isRequired,
    power: PropTypes.bool.isRequired,
};

const scannerType = PropTypes.shape(scannerShape);

const scanningJobShape = {
    disableStart: PropTypes.bool,
    duration: PropTypes.instanceOf(Duration).isRequired,
    interval: PropTypes.instanceOf(Duration).isRequired,
    name: PropTypes.string.isRequired,
    scannerId: PropTypes.string.isRequired,
};
const scanningJobType = PropTypes.shape(scanningJobShape);

const projectShape = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
};

const experimentShape = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    duration: PropTypes.number.isRequired,
    interval: PropTypes.number.isRequired,
    scannerId: PropTypes.string,
};

const newExperimentErrorsShape = {
    general: PropTypes.string,
    name: PropTypes.string,
    description: PropTypes.string,
    duration: PropTypes.string,
    interval: PropTypes.string,
    scanner: PropTypes.string,
};

export default {
    cccMetadata,
    pinningFormat,
    scannerShape,
    scannerType,
    scanningJobType,
    scanningJobShape,
    projectShape,
    experimentShape,
    newExperimentErrorsShape,
};
