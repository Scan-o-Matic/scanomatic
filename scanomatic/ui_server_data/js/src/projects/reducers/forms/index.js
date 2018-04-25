// @flow
import { combineReducers } from 'redux';
import newProject from './newProject';
import newExperiment from './newExperiment';

const forms = combineReducers({ newExperiment, newProject });

export default forms;

