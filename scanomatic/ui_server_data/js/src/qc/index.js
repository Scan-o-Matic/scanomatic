import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';
import Bridge from './bridge';
import DrawCurvesIntegration from './DrawCurvesIntegration';

const store = createStore(
    reducer,
    // { plate: {}, settings: {} },
    {},
    applyMiddleware(thunk),
);

window.qc = Bridge(store);

window.qc.subscribe(new DrawCurvesIntegration().handleUpdate);
