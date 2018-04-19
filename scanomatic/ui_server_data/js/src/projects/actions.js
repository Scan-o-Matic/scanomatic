// @flow
export type Action
    = {| type: 'NEWPROJECT_INIT' |}
    | {| type: 'NEWPROJECT_CHANGE', field: string, value: string |}
    | {| type: 'NEWPROJECT_SUBMIT' |}
    | {| type: 'NEWPROJECT_CANCEL' |}
    | {| type: 'PROJECTS_ADD', id: string, name: string, description: string |}
    | {| type: 'NEWEXPERIMENT_INIT', projectId: string |}
    | {| type: 'NEWEXPERIMENT_CHANGE', field: string, value: string |}
    | {| type: 'NEWEXPERIMENT_SUBMIT' |}
    | {| type: 'NEWEXPERIMENT_CANCEL' |}
    | {|
        type: 'EXPERIMENTS_ADD',
        description: string,
        duration: number,
        id: string,
        interval: number,
        name: string,
        projectId: string,
        scanner: string,
    |}
    | {| type: 'EXPERIMENTS_START', id: string, date: Date |}
    | {| type: 'EXPERIMENTS_STOP', id: string, date: Date |}

export function initNewProject(): Action {
    return { type: 'NEWPROJECT_INIT' };
}

export function changeNewProject(field: string, value: string): Action {
    return { type: 'NEWPROJECT_CHANGE', field, value };
}

export function submitNewProject(): Action {
    return { type: 'NEWPROJECT_SUBMIT' };
}

export function cancelNewProject(): Action {
    return { type: 'NEWPROJECT_CANCEL' };
}

export function addProject(name: string, description: string): Action {
    return {
        type: 'PROJECTS_ADD',
        name,
        description,
        id: new Date().getTime().toString(),
    };
}

export function initNewExperiment(projectId: string): Action {
    return { type: 'NEWEXPERIMENT_INIT', projectId };
}

export function changeNewExperiment(field: string, value: string): Action {
    return {
        type: 'NEWEXPERIMENT_CHANGE',
        field,
        value,
    };
}

export function submitNewExperiment(): Action {
    return {
        type: 'NEWEXPERIMENT_SUBMIT',
    };
}

export function cancelNewExperiment(): Action {
    return {
        type: 'NEWEXPERIMENT_CANCEL',
    };
}

export function addExperiment(
    projectId: string,
    name: string,
    description: string,
    duration: number,
    interval: number,
    scanner: string,
): Action {
    return {
        type: 'EXPERIMENTS_ADD',
        id: new Date().getTime().toString(),
        projectId,
        name,
        description,
        duration,
        interval,
        scanner,
    };
}

export function startExperiment(id: string): Action {
    return {
        type: 'EXPERIMENTS_START',
        id,
        date: new Date(),
    };
}

export function stopExperiment(id: string): Action {
    return {
        type: 'EXPERIMENTS_STOP',
        id,
        date: new Date(),
    };
}
