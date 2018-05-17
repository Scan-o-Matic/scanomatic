// @flow

export type Scanners = Array<{
    +name: string,
    +id: string,
    +isOnline: boolean,
    +isFree: boolean,
}>;

export type Experiments = Array<{
    +name: string,
    +description: string,
    +duration: number,
    +interval: number,
    +scannerId: string,
    +pinning: ?Array<string>,
    +started: ?Date,
    +stopped: ?Date,
    +reason: ?string,
    +done: ?boolean,
}>;

export type UpdateStatus = {
    +scanners: ?Date,
    +experiments: ?Date,
};

export type State = {
    +scanners: Scanners,
    +experiments: Experiments,
    +updateStatus: UpdateStatus,
};
