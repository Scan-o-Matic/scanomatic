import PropTypes from 'prop-types';
import React from 'react';

export default function PlateList(props) {
    return (
        <div>
          <h4>Uploaded Images</h4>
          <ul>
            {props.plates.map(({ name, id }) =>
                <li key={id}>{name}</li>
            )}
          </ul>
        </div>
    );
}

PlateList.propTypes = {
    plates: PropTypes.arrayOf(PropTypes.shape({
        name: PropTypes.string.isRequired,
        id: PropTypes.string.isRequired,
    })).isRequired,
};
