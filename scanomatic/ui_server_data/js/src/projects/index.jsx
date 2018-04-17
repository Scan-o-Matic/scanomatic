import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore } from 'redux';

import reducer from './reducers';
import ProjectsRoot from '../components/ProjectsRoot';


const store = createStore(reducer);

document.addEventListener('DOMContentLoaded', () => {
    ReactDOM.render(
        <Provider store={store}>
            <ProjectsRoot />
        </Provider>,
        document.getElementById('react-root'),
    );
});
