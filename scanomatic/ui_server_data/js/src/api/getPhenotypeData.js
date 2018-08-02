// @flow
import API from './API';

export type PlatePhenotypeData = {
    phenotypes: Array<Array<number>>, // 2d array of values
    badData: Array<Array<number>>, // [[y1, y2, ...], [x1, x2, ...]]
    empty: Array<Array<number>>, // [[y1, y2, ...], [x1, x2, ...]]
    noGrowth: Array<Array<number>>, // [[y1, y2, ...], [x1, x2, ...]]
    undecidedProblem: Array<Array<number>>, // [[y1, y2, ...], [x1, x2, ...]]
}

export default function getPhenotypeData(
    project: string,
    plate: number,
    phenotype: string,
) : Promise<PlatePhenotypeData> {
    const uri = `/api/results/phenotype/${phenotype}/${plate}/${project}`;
    return API.get(uri)
        .then(r => ({
            phenotypes: r.data,
            badData: r.BadData,
            empty: r.Empty,
            noGrowth: r.NoGrowth,
            undecidedProblem: r.UndecidedProblem,
            qIndexQueue: r.qindex_rows.map((row, idx) => ({ idx, row, col: r.qindex_cols[idx] })),
        }));
}
