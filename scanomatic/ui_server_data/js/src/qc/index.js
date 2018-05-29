import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';
import Bridge from './bridge';

const store = createStore(
    reducer,
    // { plate: {}, settings: {} },
    {},
    applyMiddleware(thunk),
);

window.qc = Bridge(store);
