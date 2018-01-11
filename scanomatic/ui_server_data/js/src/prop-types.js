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

export default {
    cccMetadata,
    pinningFormat,
};
