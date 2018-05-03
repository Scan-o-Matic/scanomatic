// @flow

export type Projects = Map<string, {
    +name: string,
    +description: string,
    +experimentIds: Array<string>,
}>;

export type Experiments = Map<string, {
    +name: string,
    +description: string,
    +duration: number,
    +interval: number,
    +scannerId: string,
    +started: ?Date,
    +stopped: ?Date,
    +reason: ?string,
}>;

export type Scanners = Map<string, {
    +name: string,
    +isOnline: boolean,
    +isFree: boolean,
}>;

export type NewProject = ?{
    +submitted: boolean,
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
