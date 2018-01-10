import PropTypes from 'prop-types';

export const scannerType = PropTypes.shape({
    name: PropTypes.string.isRequired,
    owned: PropTypes.bool.isRequired,
    power: PropTypes.bool.isRequired,
});

export const jobType = PropTypes.shape({
    duration: PropTypes.shape({
        days: PropTypes.number.isRequired,
        hours: PropTypes.number.isRequired,
        minutes: PropTypes.number.isRequired,
    }).isRequired,
    name: PropTypes.string.isRequired,
    interval: PropTypes.number.isRequired,
    scanner: scannerType.isRequired,
});
