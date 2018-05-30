// @flow
import { API } from '../api';

export const getCurveData = (project: string, plate: number, row: number, col: number) => {
    const uri = `/api/results/curves/${plate}/${row}/${col}/${project}`;
    return API.get(uri)
        .then(r => ({
            project,
            plate,
            row,
            col,
            raw: r.raw_data,
            smooth: r.smooth_data,
            times: r.time_data,
        }));
};

export default {
    getCurveData,
};
