// @flow
import { combineReducers } from 'redux';

import type { State } from '../state';
import type { Action } from '../actions';

import plate from './plate';
import settings from './settings';

const reducers: (State, Action) => State = combineReducers({ plate, settings });

export default reducers;
