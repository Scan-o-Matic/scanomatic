// @flow
import API from './API';

export type TimeSeries = Array<number>;
export type PlateOfTimeSeries = Array<Array<TimeSeries>>;

export type PlateGrowthData = {
    times: TimeSeries,
    raw: PlateOfTimeSeries,
    smooth: PlateOfTimeSeries,
};

export default function getPlateGrowthData(
    project: string,
    plate: number,
) : Promise<PlateGrowthData> {
    const uri = `/api/results/growthcurves/${plate}/${project}`;
    return API.get(uri)
        .then(r => ({
            raw: r.raw_data,
            smooth: r.smooth_data,
            times: r.times_data,
        }));
}
