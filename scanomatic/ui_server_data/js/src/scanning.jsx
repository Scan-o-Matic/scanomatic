import React from 'react';
import ReactDOM from 'react-dom';

import ScanningRootContainer from './containers/ScanningRootContainer';


$(document).ready(() => {
    ReactDOM.render(<ScanningRootContainer />, document.getElementById('react-root'));
});
