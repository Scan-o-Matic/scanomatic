import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';
import StatusPageContainer from '../containers/StatusPageContainer';

const store = createStore(
    reducer,
    {},
    applyMiddleware(thunk),
);

document.addEventListener('DOMContentLoaded', () => {
    ReactDOM.render(
        <Provider store={store}>
            <StatusPageContainer />
        </Provider>,
        document.getElementById('react-root'),
    );
});
