// @flow
import API from './API';

export type QCMarkType = 'OK' | 'NoGrowth' | 'BadData' | 'Empty' | 'UndecidedProblem';

export function setCurveQCMarkAll(
    project: string,
    plate: number,
    row: number,
    col: number,
    mark: QCMarkType,
    key: string,
) : Promise<> {
    const uri = `/api/results/curve_mark/set/${mark}/${plate}/${row}/${col}/${project}?lock_key=${key}`;
    return API.postJSON(uri, {})
        .then((r) => {
            if (!r || !r.success) throw new Error('Setting QC Mark was refused');
            return Promise.resolve();
        });
}

export function setCurveQCMark(
    project: string,
    phenotype: Phenotype,
    plate: number,
    row: number,
    col: number,
    mark: QCMarkType,
    key: string,
) : Promise<> {
    const uri = `/api/results/curve_mark/set/${mark}/${phenotype}/${plate}/${row}/${col}/${project}?lock_key=${key}`;
    return API.postJSON(uri, {})
        .then((r) => {
            if (!r || !r.success) throw new Error('Setting QC Mark was refused');
            return Promise.resolve();
        });
}
