// @flow
import { API } from '../api';

export const getPlateGrowthData = (project: string, plate: number) => {
    const uri = `/api/results/growthcurves/${plate}/${project}`;
    return API.get(uri)
        .then(r => ({
            raw: r.raw_data,
            smooth: r.smooth_data,
            times: r.times_data,
        }));
};

export default {
    getPlateGrowthData,
};
