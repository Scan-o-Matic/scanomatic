// @flow
import { combineReducers } from 'redux';
import projects from './projects';

const entities = combineReducers({ projects });
export default entities;
