// @flow
import type { State } from './state';

type Project = {
id: string,
    name: string,
    description: string,
    experiments: Array<{
        id: string,
        name: string,
        description: string,
        duration: number,
        interval: number,
        scanner: {
            name: string,
            isFree: boolean,
            isOnline: boolean,
        },
    }>,
};

export function getProjects(state: State): Array<Project> {
    return Object.keys(state.entities.projects).sort().map((key) => {
        const { name, description, experimentIds } = state.entities.projects[key];
        return {
            id: key,
            name,
            description,
            experiments: experimentIds.map((eid) => {
                const {
                    name: ename, description: edescription, duration, interval, scannerId,
                } = state.entities.experiments[eid];
                const scanner = state.entities.scanners[scannerId];
                return {
                    id: eid,
                    name: ename,
                    description: edescription,
                    duration,
                    interval,
                    scanner: { ...scanner, id: scannerId },
                };
            }),
        };
    });
}

export function getNewProject(state: State): ?{ name: string, description: string } {
    if (state.forms.newProject == null) return null;
    const { name, description } = state.forms.newProject.fields;
    return { name, description };
}

export function getNewProjectErrors(state: State): Map<string, string> {
    const { forms: { newProject } } = state;
    const errors = new Map();
    if (newProject == null) return errors;
    if (!newProject.submitted) return errors;
    if (newProject.fields.name === '') {
        errors.set('name', 'Project name cannot be empty');
    }
    return errors;
}

export function getNewExperiment(state: State): ?{
    +name: string,
    +description: string,
    +scannerId: string,
    +duration: number,
    +interval: number,
    +projectId: string,
} {
    if (state.forms.newExperiment == null) return null;
    return {
        ...state.forms.newExperiment.fields,
        projectId: state.forms.newExperiment.projectId,
    };
}

export function getNewExperimentErrors(state: State): Map<string, string> {
    const errors = new Map();
    const { forms: { newExperiment } } = state;
    if (newExperiment == null) return errors;
    if (!newExperiment.submitted) return errors;
    if (newExperiment.fields.name === '') {
        errors.set('name', 'Required');
    }
    if (newExperiment.fields.scannerId === '') {
        errors.set('scannerId', 'Required');
    }
    if (newExperiment.fields.duration < 1) {
        errors.set('duration', 'Required');
    }
    if (newExperiment.fields.interval < 1) {
        errors.set('interval', 'Required');
    }
    return errors;
}
