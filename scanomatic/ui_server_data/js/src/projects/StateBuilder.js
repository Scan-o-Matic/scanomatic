// @flow
import type { State, Projects, Experiments, Scanners, NewProject, NewExperiment } from '../../src/projects/state';

const projectDefaults = {
    name: 'Some project',
    description: 'This is a project.',
    experimentIds: [],
};

const experimentDefaults = {
    name: 'Some experiment',
    description: 'This is an experiment',
    duration: 1200,
    interval: 300,
    started: null,
    stopped: null,
    reason: null,
    scannerId: '001',
};

const scannerDefaults = {
    name: 'My scanner',
    isOnline: true,
    isFree: true,
};

type PartialExperiment = {
    id: string,
    name?: string,
    description?: string,
    duration?: number,
    interval?: number,
    scannerId?: string,
};

type PartialScanner = {
    id: string,
    name?: string,
    isOnline?: boolean,
    isFree?: boolean,
};

type PartialNewExperiment = {
    name?: string,
    description?: string,
    scannerId?: string,
    duration?: number,
    interval?: number
};

export default class StateBuilder {
    hasNewProject: boolean;
    hasNewExperiment: boolean;
    newProjectName: string;
    newProjectDescription: string;
    newProjectIsSubmitted: boolean;
    newExperimentValues: PartialNewExperiment;
    newExperimentIsSubmitted: boolean;
    newExperimentProjectId: string;
    projectIds: Array<string>;
    extraProjects: Array<{ id: string, name?: string, description?: string }>;
    experiments: Array<PartialExperiment>;
    scanners: Array<PartialScanner>;

    constructor() {
        this.hasNewExperiment = true;
        this.hasNewProject = true;
        this.newProjectName = 'A project';
        this.newProjectDescription = 'A project description';
        this.newProjectIsSubmitted = true;
        this.newExperimentValues = {};
        this.newExperimentIsSubmitted = true;
        this.newExperimentProjectId = 'P123';
        this.projectIds = ['001', '002'];
        this.extraProjects = [];
        this.experiments = [];
        this.scanners = [];
    }

    setNewProjectName(name: string): StateBuilder {
        this.newProjectName = name;
        this.hasNewProject = true;
        return this;
    }

    setNewProjectDescription(description: string): StateBuilder {
        this.newProjectDescription = description;
        this.hasNewProject = true;
        return this;
    }

    submitNewProject(): StateBuilder {
        this.newProjectIsSubmitted = true;
        return this;
    }

    unsubmitNewProject(): StateBuilder {
        this.newProjectIsSubmitted = false;
        return this;
    }

    clearNewProject() {
        this.hasNewProject = false;
        return this;
    }

    setNewExperimentValues(values: PartialNewExperiment): StateBuilder {
        this.newExperimentValues = values;
        this.hasNewExperiment = true;
        return this;
    }

    setNewExperimentProjectId(projectId: string): StateBuilder {
        this.newExperimentProjectId = projectId;
        this.hasNewExperiment = true;
        return this;
    }

    clearNewExperiment(): StateBuilder {
        this.hasNewExperiment = false;
        return this;
    }

    submitNewExperiment(): StateBuilder {
        this.newExperimentIsSubmitted = true;
        return this;
    }

    unsubmitNewExperiment(): StateBuilder {
        this.newExperimentIsSubmitted = false;
        return this;
    }

    setProjectIds(ids: Array<string>): StateBuilder {
        this.projectIds = ids;
        return this;
    }

    addProject(project: { id: string, name?: string, description?: string }): StateBuilder {
        this.extraProjects.push(project);
        return this;
    }

    addExperiment(experiment: PartialExperiment): StateBuilder {
        this.experiments.push(experiment);
        return this;
    }

    addScanner(scanner: PartialScanner): StateBuilder {
        this.scanners.push(scanner);
        return this;
    }

    clearScanners(): StateBuilder {
        this.scanners = [];
        return this;
    }

    clearProjects(): StateBuilder {
        this.extraProjects = [];
        this.projectIds = [];
        return this;
    }

    buildProjects(): Projects {
        const projects = {};
        this.projectIds.forEach((id) => {
            projects[id] = { ...projectDefaults };
        });
        this.extraProjects.forEach((p) => {
            projects[p.id] = { ...projectDefaults, ...p };
        });
        return projects;
    }

    buildExperiments(): Experiments {
        const experiments = {};
        this.experiments.forEach((e) => {
            experiments[e.id] = { ...experimentDefaults, ...e };
        });

        return experiments;
    }

    buildScanners(): Scanners {
        const scanners = {};
        this.scanners.forEach((s) => {
            scanners[s.id] = { ...scannerDefaults, ...s };
        });
        return scanners;
    }

    buildNewProject(): NewProject {
        if (!this.hasNewProject) {
            return null;
        }
        return {
            submitted: this.newProjectIsSubmitted,
            fields: {
                name: this.newProjectName,
                description: this.newProjectDescription,
            },
        };
    }

    buildNewExperiment(): NewExperiment {
        if (!this.hasNewExperiment) {
            return null;
        }
        return {
            submitted: this.newExperimentIsSubmitted,
            fields: {
                name: 'A New Xperiment',
                description: '',
                duration: 1000,
                interval: 100,
                scannerId: 'scnr01',
                ...this.newExperimentValues,
            },
            projectId: this.newExperimentProjectId,
        };
    }

    build(): State {
        return {
            entities: {
                projects: this.buildProjects(),
                experiments: this.buildExperiments(),
                scanners: this.buildScanners(),
            },
            forms: {
                newProject: this.buildNewProject(),
                newExperiment: this.buildNewExperiment(),
            },
        };
    }
}
