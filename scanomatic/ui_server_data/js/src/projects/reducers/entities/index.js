// @flow
import { combineReducers } from 'redux';
import projects from './projects';
import experiments from './experiments';
import scanners from './scanners';

const entities = combineReducers({ experiments, projects, scanners });
export default entities;
