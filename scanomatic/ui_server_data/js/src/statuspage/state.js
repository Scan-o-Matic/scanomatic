// @flow

export type Scanners = Array<{
    +name: string,
    +id: string,
    +isOnline: boolean,
}>;

export type Experiments = Array<{
    +name: string,
    +scannerId: string,
    +started: ?number,
    +stopped: ?number,
    +end: ?number,
}>;

export type UpdateStatus = {
    +scanners: boolean,
    +experiments: boolean,
};

export type State = {
    +scanners: Scanners,
    +experiments: Experiments,
    +updateStatus: UpdateStatus,
};
