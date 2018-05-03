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
            power: boolean,
            owned: boolean,
        },
    }>,
};

export function getProjects(state: State): Array<Project> {
    const projects = Array.from(
        state.entities.projects,
        ([key, { name, description, experimentIds }]) => ({
            id: key,
            name,
            description,
            experiments: experimentIds.map((eid) => {
                const experiment = state.entities.experiments.get(eid);
                if (!experiment) { throw Error(`Missing experiment with id ${eid}`); }
                const {
                    name: ename, description: edescription, duration, interval, scannerId,
                } = experiment;
                const scanner = state.entities.scanners.get(scannerId);
                if (!scanner) { throw Error(`Missing scanner with id ${scannerId}`); }
                return {
                    id: eid,
                    name: ename,
                    description: edescription,
                    duration,
                    interval,
                    scanner: {
                        id: scannerId,
                        name: scanner.name,
                        power: scanner.isOnline,
                        owned: !scanner.isFree,
                    },
                };
            }),
        }),
    );

    projects.sort((p1, p2) => {
        if (p1.id > p2.id) return -1;
        if (p1.id < p2.id) return 1;
        return 0;
    });
    return projects;
}

export function getScanners(state: State): Array<{ name: string, identifier: string, power: boolean, owned: boolean }> {
    const scanners = Array.from(
        state.entities.scanners,
        ([key, { name, isOnline, isFree }]) => (
            {
                identifier: key,
                name,
                power: isOnline,
                owned: !isFree,
            }
        ),
    );
    scanners.sort((s1, s2) => {
        if (s1.name < s2.name) return -1;
        if (s1.name > s2.name) return 1;
        return 0;
    });
    return scanners;
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
