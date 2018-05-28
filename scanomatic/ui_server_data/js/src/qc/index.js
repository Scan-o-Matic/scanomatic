import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';
import Bridge from './bridge';

const store = createStore(
    reducer,
    {},
    applyMiddleware(thunk),
);

window.qc = Bridge(store);
