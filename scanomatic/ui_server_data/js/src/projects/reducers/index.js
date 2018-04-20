// @flow
import { combineReducers } from 'redux';

import type { State } from '../state';
import type { Action } from '../actions';
import entities from './entities';
import forms from './forms';

const reducers: (State, Action) => State = combineReducers({ entities, forms });

export default reducers;
