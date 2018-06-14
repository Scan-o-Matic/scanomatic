import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';

import reducer from './reducers';

import Bridge from './bridge';
import DrawCurvesIntegration from './DrawCurvesIntegration';
import QIndexAndSelectionIntegration from './QIndexAndSelectionIntegration';

const store = createStore(
    reducer,
    {},
    applyMiddleware(thunk),
);

window.qc = Bridge(store);
window.qc.subscribe(new DrawCurvesIntegration().handleUpdate);
window.qc.subscribe(new QIndexAndSelectionIntegration(window.qc).handleUpdate);
