import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';
import StatusRootContainer from '../containers/StatusRootContainer';
import { retrieveStatus } from '../statuspage/actions';

const store = createStore(
    reducer,
    {},
    applyMiddleware(thunk),
);

const updater = () => {
    store.dispatch(retrieveStatus());
};

updater();
setInterval(updater, 300000);

document.addEventListener('DOMContentLoaded', () => {
    ReactDOM.render(
        <Provider store={store}>
            <StatusRootContainer />
        </Provider>,
        document.getElementById('react-root'),
    );
});
