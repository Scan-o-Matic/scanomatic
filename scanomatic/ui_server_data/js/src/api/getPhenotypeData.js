// @flow
import API from './API';

export type PlateValueArray = Array<Array<number>>;
export type PlateCoordinatesArray = Array<Array<number>>; // [[y1, y2, ...], [x1, x2, ...]]

export type Mark = 'badData' | 'empty' | 'noGrowth' | 'undecidedProblem';
export type QCMarksMap = Map<Mark, PlateCoordinatesArray>;

export type PlatePhenotypeData = {
    phenotypes: PlateValueArray,
    qcmarks: QCMarksMap,
    qIndexQueue: Array<{ idx: number, row: number, col: number}>,
};

export default function getPhenotypeData(
    project: string,
    plate: number,
    phenotype: string,
) : Promise<PlatePhenotypeData> {
    const uri = `/api/results/phenotype/${phenotype}/${plate}/${project}`;
    return API.get(uri)
        .then(r => ({
            phenotypes: r.data,
            qcmarks: new Map([
                ['badData', r.BadData],
                ['empty', r.Empty],
                ['noGrowth', r.NoGrowth],
                ['undecidedProblem', r.UndecidedProblem],
            ]),
            qIndexQueue: r.qindex_rows.map((row, idx) => ({ idx, row, col: r.qindex_cols[idx] })),
        }));
}
