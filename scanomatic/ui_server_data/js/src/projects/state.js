// @flow

export type Projects = {
    +[string]: {
        +name: string,
        +description: string,
        +experimentIds: Array<string>,
    }
};

export type Experiments = {
    +[string]: {
        +name: string,
        +description: string,
        +duration: number,
        +interval: number,
        +scannerId: string,
        +started: ?Date,
        +stopped: ?Date,
        +reason: ?string,
    }
};

export type Scanners = {
    +[string]: {
        +name: string,
        +isOnline: boolean,
        +isFree: boolean,
    }
};

export type Field = {
    value: string,
    touched: boolean,
};

export type NewProject = {
    +fields: {
        +name: string,
        +description: string,
    },
};

export type NewExperiment = ?{
    +projectId: string,
    +submitted: boolean,
    +fields: {
        +name: string,
        +description: string,
        +duration: number,
        +interval: number,
        +scannerId: string,
    },
};

export type State = {
    +entities: {
        +projects: Projects,
        +experiments: Experiments,
        +scanners: Scanners,
    },
    +forms: {
        +newProject: NewProject,
        +newExperiment: NewExperiment,
    },
};
