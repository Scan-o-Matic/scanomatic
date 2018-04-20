// @flow
import type { NewProject, NewExperiment } from './state';

export function validateNewProject(state: NewProject): ?Map<string, string> {
    if (state == null) return null;
    const errors = new Map();
    if (state.fields.name === '') {
        errors.set('name', 'Project name cannot be empty');
    }
    return errors;
}

export function validateNewExperiment(state: NewExperiment): ?Map<string, string> {
    if (state == null) return null;
    const errors = new Map();
    if (state.fields.name === '') {
        errors.set('name', 'Required');
    }
    if (state.fields.scannerId === '') {
        errors.set('scannerId', 'Required');
    }
    if (state.fields.duration < 1) {
        errors.set('duration', 'Required');
    }
    if (state.fields.interval < 1) {
        errors.set('interval', 'Required');
    }
    return errors;
}
