// @flow
import { combineReducers } from 'redux';

import type { State } from '../state';
import type { Action } from '../actions';

import scanners from './scanners';
import experiments from './experiments';
import updateStatus from './updateStatus';

const reducers: (State, Action) => State = combineReducers({ scanners, experiments, updateStatus });

export default reducers;
