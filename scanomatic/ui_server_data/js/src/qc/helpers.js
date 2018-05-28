// @flow
import { TimeSeries, PlateOfTimeSeries } from './state';

export const getUpdated2DArrayCopy = (
    old: PlateOfTimeSeries,
    row: number,
    col: number,
    data: TimeSeries,
) : PlateOfTimeSeries => {
    const ly = Math.max(old ? old.length : 0, row + 1);
    const lx = Math.max(old && old.length > 0 ? old[0].length : 0, col + 1);
    const arr = [];
    for (let y = 0; y < ly; y += 1) {
        const copy = old && old.length > y ? old[y].slice() : Array(lx).fill(null);
        if (y === row) {
            copy[col] = data;
        }
        arr.push(copy);
    }
    return arr;
};

export default {
    getUpdated2DArrayCopy,
};
