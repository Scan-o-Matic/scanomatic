// @flow
import { combineReducers } from 'redux';
import projects from './projects';
import experiments from './experiments';

const entities = combineReducers({ experiments, projects });
export default entities;
