import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';
import ProjectsRootContainer from '../containers/ProjectsRootContainer';


const store = createStore(reducer, applyMiddleware(thunk));

document.addEventListener('DOMContentLoaded', () => {
    ReactDOM.render(
        <Provider store={store}>
            <ProjectsRootContainer />
        </Provider>,
        document.getElementById('react-root'),
    );
});
