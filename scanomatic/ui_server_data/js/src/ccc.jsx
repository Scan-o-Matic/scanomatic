import React from 'react';
import ReactDOM from 'react-dom';

import CCCRootContainer from './containers/CCCRootContainer';


$(document).ready(() => {
    ReactDOM.render(<CCCRootContainer />, document.getElementById('react-root'));
});
