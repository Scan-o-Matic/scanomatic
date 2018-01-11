import PropTypes from 'prop-types';


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

const scannerType = PropTypes.shape({
    name: PropTypes.string.isRequired,
    owned: PropTypes.bool.isRequired,
    power: PropTypes.bool.isRequired,
});

const scanningJobType = PropTypes.shape({
    duration: PropTypes.shape({
        days: PropTypes.number.isRequired,
        hours: PropTypes.number.isRequired,
        minutes: PropTypes.number.isRequired,
    }).isRequired,
    name: PropTypes.string.isRequired,
    interval: PropTypes.number.isRequired,
    scanner: scannerType.isRequired,
});

export default {
    cccMetadata,
    pinningFormat,
    scannerType,
    scanningJobType,
};
